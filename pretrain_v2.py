#!/usr/bin/env python3
"""
Geometry Transfer, Stage 1 (v2): self-supervised point-cloud pretraining on ShapeNet.

Implements the proposal's Stage 1 as written, with the problems found in review fixed:

  Proposal                                   | v2 implementation
  -------------------------------------------|---------------------------------------------
  PointNet++-style set-abstraction backbone  | FPS + kNN/ball-query grouping, 2 local SA
                                             | levels + global SA (was: per-point MLP + max)
  Masked completion, Chamfer on missing part | contiguous patch removed, decoder predicts ONLY
                                             | the hidden patch, Chamfer vs hidden points
  Rotation-invariant contrastive, two        | two independently rotated views per shape,
  rotated views, other shapes as negatives   | per-sample rotations, projection head with no
                                             | final ReLU, symmetric InfoNCE, other shapes
                                             | in the batch are the negatives
  Regular checkpointing, partial results OK  | atomic latest.pth every epoch, --resume,
                                             | --max-hours clean stop
  ~1.5-3.5M parameter encoder                | printed at startup

Added from the review:
  * frozen-encoder linear probe on HELD-OUT ShapeNet shapes, run every few epochs,
    under clean / yaw / occlusion / noise / clutter / all-combined conditions
  * best.pth is chosen by that probe, never by SSL loss (loss and probe disagreed in review)
  * random-init probe baseline at epoch 0, and a loud warning if SSL ends up below it
  * corruption-aware augmentations are OPT-IN and separable (--aug-corrupt occ,noise,clutter)
    so each factor can be ablated on its own run

Usage (inside the rocm container, inside tmux):
  python3 pretrain_v2.py prepare                       # one-time, reuses the old cache if valid
  python3 pretrain_v2.py train --synthetic --epochs 5  # 2-minute pipeline sanity check
  python3 pretrain_v2.py train --epochs 2 --probe-every 1   # timing trial on real data
  python3 pretrain_v2.py train --epochs 150 --seed 0
  python3 pretrain_v2.py probe --ckpt <work>/runs/<name>/best.pth   # or --ckpt none

Stage 2 loads the encoder with:  from pretrain_v2 import load_encoder
"""

import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_POINTS = 1024

# Probe conditions. Levels are fixed so the corrupted validation sets are identical
# at every epoch and across runs; training augmentation draws from TRAIN_RANGES
# instead, so the probe is not simply the training distribution.
CONDITIONS = ["clean", "yaw", "occ", "noise", "clutter", "scan"]
EVAL_LEVELS = dict(occ=0.30, clutter=0.15, noise=0.02)
TRAIN_RANGES = dict(occ=(0.10, 0.40), clutter=(0.05, 0.25), noise=(0.005, 0.03))

SYNSET_NAMES = {
    "02691156": "airplane", "02828884": "bench", "02933112": "cabinet", "02958343": "car",
    "03001627": "chair", "03211117": "display", "03636649": "lamp", "03691459": "loudspeaker",
    "04090263": "rifle", "04256520": "sofa", "04379243": "table", "04401088": "telephone",
    "04530566": "watercraft",
}

_LOG_PATH = None


def log(msg):
    print(msg, flush=True)
    if _LOG_PATH:
        with open(_LOG_PATH, "a") as f:
            f.write(msg + "\n")


def first_existing(cands):
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


# =============================================================================
# 1. Data preparation (CPU only, one time)
# =============================================================================

def list_meshes(data_dir):
    """Same glob as the original script, unsorted on purpose: the old cache files are
    indexed by this order, so sorting here would silently misalign them."""
    files = glob.glob(os.path.join(data_dir, "**", "models", "model_normalized.obj"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(data_dir, "**", "*.obj"), recursive=True)
    return files


def synset_of(path):
    for part in reversed(Path(path).parts[:-1]):
        if re.fullmatch(r"\d{8}", part):
            return part
    return None


def fps_numpy(points, n, rng):
    N = len(points)
    idx = np.zeros(n, dtype=np.int64)
    dist = np.full(N, 1e10)
    far = rng.randint(0, N)
    for i in range(n):
        idx[i] = far
        dist = np.minimum(dist, ((points - points[far]) ** 2).sum(1))
        far = int(dist.argmax())
    return points[idx]


def normalize_np(p):
    p = p - p.mean(0)
    return p / np.linalg.norm(p, axis=1).max()


def sample_mesh_cloud(path, n_points=N_POINTS, seed=0):
    import trimesh
    np.random.seed(seed)  # trimesh samples from the global numpy RNG
    mesh = trimesh.load(path, force="mesh")
    pts = np.asarray(mesh.sample(n_points * 4))
    return normalize_np(fps_numpy(pts, n_points, np.random.RandomState(seed))).astype(np.float32)


def _worker(args):
    i, path = args
    try:
        return i, sample_mesh_cloud(path, seed=i)
    except Exception as e:  # noqa: BLE001
        return i, f"{type(e).__name__}: {e}"


def chamfer_np(a, b):
    d = ((a[:, None] - b[None]) ** 2).sum(-1)
    return d.min(1).mean() + d.min(0).mean()


def verify_alignment(files, cache_dir, k, seed=0):
    """The old cache is indexed by glob order. Before trusting it for labels, re-sample a few
    meshes and check each one is closer to ITS cached cloud than to random other clouds."""
    rng = np.random.RandomState(seed)
    ids = rng.choice(len(files), min(k, len(files)), replace=False)
    hits = 0
    for i in ids:
        ref = np.load(os.path.join(cache_dir, f"{i}.npy"))
        fresh = sample_mesh_cloud(files[i], seed=12345)
        d_self = chamfer_np(fresh, ref)
        others = [j for j in rng.choice(len(files), 6, replace=False) if j != i][:5]
        d_oth = min(chamfer_np(fresh, np.load(os.path.join(cache_dir, f"{j}.npy"))) for j in others)
        ok = d_self < d_oth
        hits += int(ok)
        log(f"  align check idx {i:6d}: self {d_self:.4f} vs best-other {d_oth:.4f} -> {'ok' if ok else 'MISMATCH'}")
    return hits >= len(ids) - 1


def cmd_prepare(args):
    data_dir = args.data_dir or first_existing(["/root/exea-project/shapenet_full", "/root/shapenet_full"])
    cache_dir = args.cache_dir or first_existing(["/root/exea-project/point_cache", "/root/point_cache"])
    out = Path(args.work_dir) / "data"
    out.mkdir(parents=True, exist_ok=True)
    global _LOG_PATH
    _LOG_PATH = str(out / "prepare_log.txt")

    files = list_meshes(data_dir)
    log(f"data dir: {data_dir} | meshes found: {len(files)}")
    if not files:
        sys.exit("No .obj files found. Pass --data-dir.")

    use_cache = not args.rebuild
    if use_cache:
        have = set(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else set()
        missing = sum(1 for i in range(len(files)) if f"{i}.npy" not in have)
        if missing:
            log(f"old cache at {cache_dir} is missing {missing} files -> rebuilding from meshes")
            use_cache = False
        else:
            log(f"old cache found at {cache_dir}; verifying it lines up with the mesh list...")
            if not verify_alignment(files, cache_dir, args.verify_n):
                sys.exit("Old cache does NOT align with the current mesh order (labels would be wrong).\n"
                         "Re-run with --rebuild (re-samples every mesh, parallel).")
            log("old cache verified")

    n = len(files)
    points = np.zeros((n, N_POINTS, 3), dtype=np.float32)
    ok = np.ones(n, dtype=bool)
    if use_cache:
        order = files
        for i in range(n):
            points[i] = np.load(os.path.join(cache_dir, f"{i}.npy"))
            if i % 10000 == 0:
                log(f"  loaded {i}/{n}")
    else:
        order = sorted(files)
        t0 = time.time()
        with Pool(args.workers) as pool:
            for done, (i, res) in enumerate(pool.imap_unordered(_worker, list(enumerate(order)), chunksize=8)):
                if isinstance(res, str):
                    ok[i] = False
                    log(f"  FAILED {order[i]}: {res}")
                else:
                    points[i] = res
                if done % 2000 == 0:
                    log(f"  sampled {done}/{n} ({time.time() - t0:.0f}s)")

    # Drop unusable clouds explicitly (the old script silently substituted zeros).
    norms = np.linalg.norm(points, axis=2).max(1)
    ok &= np.isfinite(points).all(axis=(1, 2)) & (norms > 0.9) & (norms < 1.1)
    log(f"valid clouds: {int(ok.sum())}/{n} (dropped {int((~ok).sum())})")

    synsets = [synset_of(f) for f in order]
    names = sorted({s for s in synsets if s})
    labels = np.array([names.index(s) if s else -1 for s in synsets], dtype=np.int64)
    points, labels = points[ok], labels[ok]
    kept_files = [f for f, o in zip(order, ok) if o]

    np.save(out / "shapenet_1024.npy", points)
    np.save(out / "shapenet_labels.npy", labels)
    (out / "shapenet_files.txt").write_text("\n".join(kept_files))
    meta = dict(synsets=names, counts={s: int((labels == i).sum()) for i, s in enumerate(names)},
                unlabeled=int((labels < 0).sum()), n=int(len(points)),
                source="old_cache" if use_cache else "rebuild")
    (out / "shapenet_meta.json").write_text(json.dumps(meta, indent=1))
    log(f"saved {points.shape} to {out} | categories: {len(names)} | unlabeled: {meta['unlabeled']}")


# =============================================================================
# 2. Synthetic corpus (pipeline sanity check only, NOT a benchmark)
# =============================================================================

def make_synthetic(n_per_class, seed, n_points=N_POINTS):
    rng = np.random.default_rng(seed)

    def sphere(n):
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def cube(n):
        face = rng.integers(0, 6, n)
        p = rng.uniform(-1, 1, (n, 3))
        p[np.arange(n), face // 2] = (face % 2) * 2 - 1
        return p

    def cylinder(n):
        th = rng.uniform(0, 2 * np.pi, n)
        side = rng.random(n) < 0.7
        r = np.where(side, 0.6, 0.6 * np.sqrt(rng.random(n)))
        z = np.where(side, rng.uniform(-1, 1, n), np.sign(rng.normal(size=n)))
        return np.stack([r * np.cos(th), r * np.sin(th), z], 1)

    def cone(n):
        th, t = rng.uniform(0, 2 * np.pi, n), np.sqrt(rng.random(n))
        return np.stack([0.7 * t * np.cos(th), 0.7 * t * np.sin(th), 1.2 * (1 - t) - 0.6], 1)

    def torus(n):
        u, v = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
        return np.stack([(0.75 + 0.3 * np.cos(v)) * np.cos(u), (0.75 + 0.3 * np.cos(v)) * np.sin(u),
                         0.3 * np.sin(v)], 1)

    pts, labels = [], []
    for c, gen in enumerate([sphere, cube, cylinder, cone, torus]):
        for _ in range(n_per_class):
            p = gen(n_points) * rng.uniform(0.6, 1.4, 3)
            th = rng.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
            pts.append(normalize_np(p @ R.T).astype(np.float32))
            labels.append(c)
    return np.stack(pts), np.array(labels, dtype=np.int64), ["sphere", "cube", "cylinder", "cone", "torus"]


# =============================================================================
# 3. Model: PointNet++-style encoder (FPS + local grouping), decoder, projection head
# =============================================================================

def index_points(x, idx):
    """x: (B,N,C); idx: (B,S) or (B,S,K) -> (B,S,C) or (B,S,K,C)"""
    B = x.shape[0]
    shape = [B] + [1] * (idx.dim() - 1)
    b = torch.arange(B, device=x.device).view(shape).expand_as(idx)
    return x[b, idx]


@torch.no_grad()
def farthest_point_sample(xyz, npoint, gen=None):
    B, N, _ = xyz.shape
    cent = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float("inf"), device=xyz.device)
    far = torch.randint(0, N, (B,), device=xyz.device, generator=gen)
    bidx = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        cent[:, i] = far
        d = ((xyz - xyz[bidx, far].unsqueeze(1)) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return cent


class SharedMLP(nn.Module):
    """Point-wise MLP (Linear-BN-ReLU stack) applied over the last dim of any tensor."""

    def __init__(self, dims):
        super().__init__()
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b, bias=False), nn.BatchNorm1d(b), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        shp = x.shape[:-1]
        return self.net(x.reshape(-1, x.shape[-1])).reshape(*shp, -1)


class SetAbstraction(nn.Module):
    """FPS picks `npoint` centers; each center gathers `k` neighbours (kNN, or ball query
    with radius fallback to the nearest point); a shared MLP runs on neighbour offsets
    (+ neighbour features); max over the neighbourhood."""

    def __init__(self, npoint, k, radius, in_ch, mlp, grouping):
        super().__init__()
        self.npoint, self.k, self.radius, self.grouping = npoint, k, radius, grouping
        self.mlp = SharedMLP([in_ch + 3] + mlp)

    def forward(self, xyz, feats, gen=None):
        B, N, _ = xyz.shape
        S, K = min(self.npoint, N), min(self.k, N)
        with torch.no_grad():
            new_xyz = index_points(xyz, farthest_point_sample(xyz, S, gen))
            dk, idx = torch.cdist(new_xyz, xyz).topk(K, dim=-1, largest=False)
            if self.grouping == "ball":
                idx = torch.where(dk > self.radius, idx[..., :1].expand_as(idx), idx)
        rel = (index_points(xyz, idx) - new_xyz.unsqueeze(2)) / self.radius
        g = rel if feats is None else torch.cat([rel, index_points(feats, idx)], -1)
        return new_xyz, self.mlp(g).amax(2)


class GlobalAbstraction(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()
        self.mlp = SharedMLP([in_ch + 3] + mlp)

    def forward(self, xyz, feats):
        return self.mlp(torch.cat([xyz, feats], -1)).amax(1)


class Encoder(nn.Module):
    def __init__(self, grouping="knn"):
        super().__init__()
        self.cfg = dict(grouping=grouping)
        self.sa1 = SetAbstraction(512, 32, 0.2, 0, [96, 96, 192], grouping)
        self.sa2 = SetAbstraction(128, 64, 0.4, 192, [192, 192, 384], grouping)
        self.glob = GlobalAbstraction(384, [512, 1024, 1024])
        self.out_dim = 1024

    def forward(self, xyz, gen=None):
        xyz1, f1 = self.sa1(xyz, None, gen)
        xyz2, f2 = self.sa2(xyz1, f1, gen)
        return self.glob(xyz2, f2)


class Decoder(nn.Module):
    """Predicts ONLY the hidden patch (n_out points), not the whole cloud."""

    def __init__(self, emb_dim, n_out):
        super().__init__()
        self.n_out = n_out
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 1024), nn.ReLU(inplace=True),
                                 nn.Linear(1024, 1024), nn.ReLU(inplace=True),
                                 nn.Linear(1024, n_out * 3))

    def forward(self, z):
        return self.mlp(z).view(-1, self.n_out, 3)


class ProjectionHead(nn.Module):
    """The encoder ends in ReLU+max, so its output is >= 0 and all cosines are >= 0, which puts
    a floor under InfoNCE. The final Linear here has no ReLU, so the loss lives in a space
    where negatives can actually be pushed apart."""

    def __init__(self, dim, hidden=512, out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def load_encoder(path, device="cpu"):
    """For Stage 2. Accepts best.pth / latest.pth / encoder_*.pth."""
    ck = torch.load(path, map_location=device)
    enc = Encoder(**ck["encoder_cfg"]).to(device)
    enc.load_state_dict(ck["encoder"])
    return enc


# =============================================================================
# 4. Losses, masking, augmentation
# =============================================================================

def chamfer(pred, target):
    """Squared-L2 Chamfer, both directions, explicit (gradient-safe) form."""
    d = ((pred.unsqueeze(2) - target.unsqueeze(1)) ** 2).sum(-1)
    return d.min(2)[0].mean() + d.min(1)[0].mean()


def info_nce(a, b, tau):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    logits = a @ b.T / tau
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def split_mask(points, ratio, gen=None):
    """Remove one contiguous patch (the `ratio` fraction of points nearest a random anchor,
    chosen independently per shape). Returns (visible, hidden)."""
    B, N, _ = points.shape
    n_mask = int(N * ratio)
    ai = torch.randint(0, N, (B,), device=points.device, generator=gen)
    anchor = points[torch.arange(B, device=points.device), ai].unsqueeze(1)
    order = ((points - anchor) ** 2).sum(-1).argsort(1)
    take = lambda idx: torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))  # noqa: E731
    return take(order[:, n_mask:]), take(order[:, :n_mask])


def random_rotation(B, mode, device, gen=None):
    if mode == "z":
        th = torch.rand(B, device=device, generator=gen) * (2 * math.pi)
        c, s = th.cos(), th.sin()
        R = torch.zeros(B, 3, 3, device=device)
        R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1.0
        return R
    q = torch.randn(B, 4, device=device, generator=gen)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1)], 1)


def perturb(p, gen=None, rot=None, occ=None, clutter=None, noise=None):
    """Independent per-sample perturbations. rot: None|'z'|'so3'. occ/clutter/noise: None or a
    (B,) tensor of levels (fraction removed / fraction replaced by background / sigma)."""
    B, N, _ = p.shape
    dev = p.device
    if rot:
        p = p @ random_rotation(B, rot, dev, gen).transpose(1, 2)
    changed = False
    if occ is not None:  # contiguous occlusion, then resample survivors back up to N points
        ai = torch.randint(0, N, (B,), device=dev, generator=gen)
        anchor = p[torch.arange(B, device=dev), ai].unsqueeze(1)
        rank = ((p - anchor) ** 2).sum(-1).argsort(1).argsort(1)
        keep = (rank >= (occ * N).long().unsqueeze(1)).float()
        sel = torch.multinomial(keep, N, replacement=True, generator=gen)
        p = torch.gather(p, 1, sel.unsqueeze(-1).expand(-1, -1, 3))
        changed = True
    if clutter is not None:  # background points from a surrounding box
        m = torch.rand(B, N, device=dev, generator=gen) < clutter.unsqueeze(1)
        bg = torch.rand(B, N, 3, device=dev, generator=gen) * 2.4 - 1.2
        p = torch.where(m.unsqueeze(-1), bg, p)
        changed = True
    if noise is not None:
        p = p + torch.randn(p.shape, device=dev, generator=gen) * noise.view(B, 1, 1)
        changed = True
    if changed:  # same re-normalisation a real scan pipeline applies
        p = p - p.mean(1, keepdim=True)
        p = p / p.norm(dim=-1).amax(1).clamp_min(1e-6).view(B, 1, 1)
    return p


def train_view(p, rot, corrupt, gen=None):
    B, dev = p.size(0), p.device
    lv = {}
    for k in corrupt:
        lo, hi = TRAIN_RANGES[k]
        lv[k] = lo + (hi - lo) * torch.rand(B, device=dev, generator=gen)
    return perturb(p, gen, rot=rot, **lv)


def base_augment(p, jitter=0.01, clip=0.03):
    s = 0.9 + 0.2 * torch.rand(p.size(0), 1, 1, device=p.device)
    return (p + (torch.randn_like(p) * jitter).clamp(-clip, clip)) * s


def apply_condition(p, cond, gen, rot_mode):
    B, dev = p.size(0), p.device
    full = lambda v: torch.full((B,), v, device=dev)  # noqa: E731
    spec = {"clean": {}, "yaw": dict(rot=rot_mode), "occ": dict(occ=full(EVAL_LEVELS["occ"])),
            "noise": dict(noise=full(EVAL_LEVELS["noise"])),
            "clutter": dict(clutter=full(EVAL_LEVELS["clutter"])),
            "scan": dict(rot=rot_mode, occ=full(EVAL_LEVELS["occ"]), clutter=full(EVAL_LEVELS["clutter"]),
                         noise=full(EVAL_LEVELS["noise"]))}[cond]
    return perturb(p, gen, **spec)


# =============================================================================
# 5. Frozen-encoder linear probe on held-out shapes
# =============================================================================

def stratified_split(labels, val_frac, seed):
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for lab in np.unique(labels):
        idx = rng.permutation(np.where(labels == lab)[0])
        nv = int(round(len(idx) * val_frac)) if len(idx) >= 20 else 0
        va.append(idx[:nv])
        tr.append(idx[nv:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def build_probe(labels, names, tr_idx, va_idx, k, per_class, seed, dev):
    rng = np.random.RandomState(seed)
    lab_tr = labels[tr_idx]
    cnt = {l: int((lab_tr == l).sum()) for l in np.unique(lab_tr) if l >= 0}
    top = sorted(cnt, key=lambda l: -cnt[l])[:k]
    p_tr, y_tr = [], []
    for ci, l in enumerate(top):
        pool = tr_idx[lab_tr == l]
        take = rng.choice(pool, min(per_class, len(pool)), replace=False)
        p_tr.append(take)
        y_tr.append(np.full(len(take), ci))
    va = va_idx[np.isin(labels[va_idx], top)]
    remap = {l: i for i, l in enumerate(top)}
    y_va = np.array([remap[l] for l in labels[va]])
    T = lambda a: torch.as_tensor(a, dtype=torch.long, device=dev)  # noqa: E731
    log("probe classes: " + ", ".join(f"{names[l]}" for l in top) +
        f" | probe-train {sum(len(x) for x in p_tr)} | held-out val {len(va)}")
    return dict(tr=T(np.concatenate(p_tr)), ytr=T(np.concatenate(y_tr)), va=T(va), yva=T(y_va), k=len(top))


@torch.no_grad()
def extract_features(enc, data, idx, cond, seed, rot_mode, bs=256):
    was = enc.training
    enc.eval()
    dev = data.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    out = [enc(apply_condition(data[idx[i:i + bs]], cond, gen, rot_mode), gen) for i in range(0, len(idx), bs)]
    enc.train(was)
    return torch.cat(out)


def run_probe(enc, data, probe, rot_mode, seed=0):
    dev = data.device
    Xtr = extract_features(enc, data, probe["tr"], "clean", seed, rot_mode)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xn = (Xtr - mu) / sd
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.manual_seed(seed)
        clf = nn.Linear(Xn.size(1), probe["k"]).to(dev)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
        for _ in range(300):
            opt.zero_grad()
            F.cross_entropy(clf(Xn), probe["ytr"]).backward()
            opt.step()
    res = {}
    with torch.no_grad():
        for c in CONDITIONS:
            Xv = (extract_features(enc, data, probe["va"], c, seed + 1, rot_mode) - mu) / sd
            pred = clf(Xv).argmax(1)
            oa = (pred == probe["yva"]).float().mean().item() * 100
            rec = [(pred[probe["yva"] == k] == k).float().mean().item() for k in range(probe["k"])
                   if (probe["yva"] == k).any()]
            res[c] = dict(oa=oa, mca=100 * float(np.mean(rec)))
    return res


def probe_score(res, how):
    if how == "clean":
        return res["clean"]["mca"]
    if how == "scan":
        return res["scan"]["mca"]
    return 0.5 * (res["clean"]["mca"] + res["scan"]["mca"])


def fmt_probe(res):
    return "  ".join(f"{c}:{res[c]['mca']:5.1f}" for c in CONDITIONS)


@torch.no_grad()
def embedding_stats(enc, data, idx, n=256):
    was = enc.training
    enc.eval()
    e = enc(data[idx[:n]])
    z = F.normalize(e, dim=1)
    sim = z @ z.T
    off = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    enc.train(was)
    return sim[off].mean().item(), e.std(0).mean().item()


# =============================================================================
# 6. Training
# =============================================================================

def load_data(args, dev):
    if args.synthetic:
        pts, labels, names = make_synthetic(args.synthetic_n, args.seed, args.synthetic_points)
    else:
        d = Path(args.work_dir) / "data"
        if not (d / "shapenet_1024.npy").exists():
            sys.exit(f"{d}/shapenet_1024.npy not found. Run: python3 pretrain_v2.py prepare")
        pts, labels = np.load(d / "shapenet_1024.npy"), np.load(d / "shapenet_labels.npy")
        meta = json.loads((d / "shapenet_meta.json").read_text())
        names = [SYNSET_NAMES.get(s, s) for s in meta["synsets"]]
    return torch.from_numpy(pts).to(dev), labels, names


def args_dict(args):
    return {k: v for k, v in vars(args).items() if k != "fn"}


def save_ckpt(path, enc, dec, proj, opt, sched, epoch, best, args, extra=None):
    d = dict(encoder=enc.state_dict(), encoder_cfg=enc.cfg, decoder=dec.state_dict(), proj=proj.state_dict(),
             opt=opt.state_dict(), sched=sched.state_dict(), epoch=epoch, best=best, args=args_dict(args))
    d.update(extra or {})
    tmp = str(path) + ".tmp"
    torch.save(d, tmp)
    os.replace(tmp, path)  # atomic: a disconnect mid-write cannot corrupt the previous file


def export_encoder(path, enc, epoch, probe_res):
    torch.save(dict(encoder=enc.state_dict(), encoder_cfg=enc.cfg, epoch=epoch, probe=probe_res), path)


def cmd_train(args):
    global _LOG_PATH
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corrupt = [c for c in args.aug_corrupt.split(",") if c]
    assert all(c in TRAIN_RANGES for c in corrupt), f"--aug-corrupt takes a subset of {list(TRAIN_RANGES)}"
    name = args.run_name or f"s{args.seed}_rot{args.rot}" + ("_" + "+".join(corrupt) if corrupt else "")
    run = Path(args.work_dir) / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = str(run / "train_log.txt")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    log(f"=== run {name} | device {dev}"
        f"{' | ' + torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''} | torch {torch.__version__}"
        f"{' | hip ' + str(torch.version.hip) if getattr(torch.version, 'hip', None) else ''}")

    data, labels, names = load_data(args, dev)
    N = data.shape[1]
    tr_idx, va_idx = stratified_split(labels, args.val_frac, args.seed)
    log(f"shapes: {len(data)} | SSL-train {len(tr_idx)} | held-out {len(va_idx)} | points/shape {N}")
    probe = build_probe(labels, names, tr_idx, va_idx, args.probe_classes, args.probe_per_class, args.seed, dev)
    tr_t = torch.as_tensor(tr_idx, dtype=torch.long, device=dev)
    va_t = torch.as_tensor(va_idx, dtype=torch.long, device=dev)

    n_mask = int(N * args.mask_ratio)
    enc, proj = Encoder(args.grouping).to(dev), None
    dec = Decoder(enc.out_dim, n_mask).to(dev)
    proj = ProjectionHead(enc.out_dim).to(dev)
    log(f"params: encoder {n_params(enc) / 1e6:.2f}M | decoder {n_params(dec) / 1e6:.2f}M | "
        f"proj {n_params(proj) / 1e6:.2f}M | grouping {args.grouping} | hidden patch {n_mask} pts")
    log(f"objective: {args.w_rec}*chamfer(hidden patch) + {args.w_con}*InfoNCE(tau={args.temp}) | "
        f"rotation {args.rot} | corruption aug: {corrupt or 'none'} | batch {args.batch} "
        f"(InfoNCE at random-init ~ ln({args.batch})={math.log(args.batch):.2f})")

    params = list(enc.parameters()) + list(dec.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps = len(tr_idx) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps,
                                                pct_start=0.05)
    (run / "config.json").write_text(json.dumps(args_dict(args), indent=1))

    start, best = 0, dict(score=-1.0, epoch=0, probe=None)
    hist = run / "metrics.jsonl"
    if args.resume and (run / "latest.pth").exists():
        ck = torch.load(run / "latest.pth", map_location=dev)
        for k in ("epochs", "batch", "lr", "seed"):
            if ck["args"].get(k) != args_dict(args).get(k):
                sys.exit(f"--resume needs identical {k} (saved {ck['args'].get(k)}, now {args_dict(args).get(k)})")
        enc.load_state_dict(ck["encoder"])
        dec.load_state_dict(ck["decoder"])
        proj.load_state_dict(ck["proj"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"], ck["best"]
        log(f"resumed from epoch {start} (best so far {best['score']:.1f} @ epoch {best['epoch']})")
        base = ck.get("base_probe")
    else:
        base = None
    if base is None:
        base = run_probe(enc, data, probe, args.rot, args.seed)
        log(f"[random-init probe, mean-class-acc %]  {fmt_probe(base)}")
    base_score = probe_score(base, args.select)
    last_probe = None
    t_start, bad, ep = time.time(), 0, start

    for ep in range(start + 1, args.epochs + 1):
        enc.train(), dec.train(), proj.train()
        t0 = time.time()
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        acc_r = acc_c = torch.zeros((), device=dev)
        n_ok = 0
        for s in range(steps):
            p = base_augment(data[perm[s * args.batch:(s + 1) * args.batch]])
            vis, hidden = split_mask(p, args.mask_ratio)
            loss_r = chamfer(dec(enc(vis)), hidden)
            h = enc(torch.cat([train_view(p, args.rot, corrupt), train_view(p, args.rot, corrupt)], 0))
            loss_c = info_nce(proj(h[:len(p)]), proj(h[len(p):]), args.temp)
            loss = args.w_rec * loss_r + args.w_con * loss_c
            if not torch.isfinite(loss):
                bad += 1
                opt.zero_grad(set_to_none=True)
                if bad > 20:
                    sys.exit("Loss non-finite for 20 steps in a row; aborting. latest.pth is the last good epoch.")
                continue
            bad = 0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            sched.step()
            acc_r, acc_c, n_ok = acc_r + loss_r.detach(), acc_c + loss_c.detach(), n_ok + 1
        n_ok = max(n_ok, 1)
        cos, fstd = embedding_stats(enc, data, va_t)
        mem = f" | peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if dev.type == "cuda" else ""
        rec = dict(epoch=ep, rec=acc_r.item() / n_ok, con=acc_c.item() / n_ok, cos=cos, feat_std=fstd,
                   lr=sched.get_last_lr()[0], sec=time.time() - t0)
        log(f"ep {ep:3d}/{args.epochs} | rec {rec['rec']:.4f} | con {rec['con']:.3f} | cos {cos:.2f} | "
            f"lr {rec['lr']:.1e} | {rec['sec']:.0f}s{mem}")

        timed_out = args.max_hours > 0 and (time.time() - t_start) / 3600 > args.max_hours
        if ep % args.probe_every == 0 or ep == args.epochs or timed_out:
            last_probe = run_probe(enc, data, probe, args.rot, args.seed)
            sc = probe_score(last_probe, args.select)
            rec["probe"] = last_probe
            tag = ""
            if sc > best["score"]:
                best = dict(score=sc, epoch=ep, probe=last_probe)
                export_encoder(run / "best.pth", enc, ep, last_probe)
                tag = "  <- best"
            log(f"   [probe mCA %] {fmt_probe(last_probe)} | select({args.select}) {sc:.1f} "
                f"(random-init {base_score:.1f}){tag}")
        with open(hist, "a") as f:
            f.write(json.dumps(rec) + "\n")
        save_ckpt(run / "latest.pth", enc, dec, proj, opt, sched, ep, best, args, dict(base_probe=base))
        if args.save_every and ep % args.save_every == 0:
            export_encoder(run / f"encoder_ep{ep:03d}.pth", enc, ep, last_probe)
        if timed_out:
            log(f"--max-hours reached; stopped cleanly after epoch {ep}. Resume with --resume.")
            break

    if last_probe is None:
        last_probe = run_probe(enc, data, probe, args.rot, args.seed)
    export_encoder(run / "final.pth", enc, ep, last_probe)
    log("\n=== summary (mean class accuracy %, linear probe on held-out shapes; clean-trained head) ===")
    log(f"{'':14s}" + "".join(f"{c:>9s}" for c in CONDITIONS))
    for lab, r in [("random-init", base), (f"best ep{best['epoch']}", best["probe"]), (f"final ep{ep}", last_probe)]:
        if r:
            log(f"{lab:14s}" + "".join(f"{r[c]['mca']:9.1f}" for c in CONDITIONS))
    if best["score"] < base_score:
        log("WARNING: no checkpoint beat the random-init encoder on the selection metric. SSL is not "
            "helping in this configuration; do not start Stage 2 from these weights.")
    if last_probe["clean"]["mca"] < base["clean"]["mca"] - 2:
        log("WARNING: final clean probe accuracy is below random-init; the objective is degrading the encoder.")
    (run / "summary.json").write_text(json.dumps(dict(random_init=base, best=best, final=last_probe,
                                                       params_encoder=n_params(enc)), indent=1))
    log(f"Stage 2 should load: {run / 'best.pth'}   (selected by probe, not by loss)")


def cmd_probe(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, labels, names = load_data(args, dev)
    tr_idx, va_idx = stratified_split(labels, args.val_frac, args.seed)
    probe = build_probe(labels, names, tr_idx, va_idx, args.probe_classes, args.probe_per_class, args.seed, dev)
    torch.manual_seed(args.seed)
    enc = Encoder(args.grouping).to(dev) if args.ckpt == "none" else load_encoder(args.ckpt, dev)
    res = run_probe(enc, data, probe, args.rot, args.seed)
    log(f"{args.ckpt}: {fmt_probe(res)}")


# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--work-dir", default="stage1_v2")
        p.add_argument("--seed", type=int, default=0)

    pp = sub.add_parser("prepare")
    common(pp)
    pp.add_argument("--data-dir")
    pp.add_argument("--cache-dir")
    pp.add_argument("--rebuild", action="store_true")
    pp.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    pp.add_argument("--verify-n", type=int, default=8)
    pp.set_defaults(fn=cmd_prepare)

    def probe_args(p):
        p.add_argument("--synthetic", action="store_true", help="procedural shapes, pipeline sanity check only")
        p.add_argument("--synthetic-n", type=int, default=400)
        p.add_argument("--synthetic-points", type=int, default=N_POINTS)
        p.add_argument("--val-frac", type=float, default=0.05)
        p.add_argument("--probe-classes", type=int, default=13)
        p.add_argument("--probe-per-class", type=int, default=300)
        p.add_argument("--select", choices=["mean", "clean", "scan"], default="mean")
        p.add_argument("--grouping", choices=["knn", "ball"], default="knn")
        p.add_argument("--rot", choices=["z", "so3"], default="z")

    pt = sub.add_parser("train")
    common(pt)
    probe_args(pt)
    pt.add_argument("--run-name")
    pt.add_argument("--epochs", type=int, default=150)
    pt.add_argument("--batch", type=int, default=128)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--wd", type=float, default=1e-4)
    pt.add_argument("--clip", type=float, default=5.0)
    pt.add_argument("--temp", type=float, default=0.1)
    pt.add_argument("--mask-ratio", type=float, default=0.25)
    pt.add_argument("--w-rec", type=float, default=1.0)
    pt.add_argument("--w-con", type=float, default=1.0)
    pt.add_argument("--aug-corrupt", default="", help="subset of occ,noise,clutter (default: rotation only)")
    pt.add_argument("--probe-every", type=int, default=5)
    pt.add_argument("--save-every", type=int, default=25)
    pt.add_argument("--max-hours", type=float, default=0.0)
    pt.add_argument("--resume", action="store_true")
    pt.set_defaults(fn=cmd_train)

    pq = sub.add_parser("probe")
    common(pq)
    probe_args(pq)
    pq.add_argument("--ckpt", required=True, help="path, or 'none' for a random-init encoder")
    pq.set_defaults(fn=cmd_probe)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
