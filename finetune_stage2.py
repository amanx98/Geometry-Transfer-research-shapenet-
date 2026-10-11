#!/usr/bin/env python3
"""
Stage 2: sim-to-real transfer experiment. This is the proposal's CORE result -- the delta
between condition 2 and condition 3 below.

REFUSES TO RUN until dataset_report.json (from inspect_data.py) exists and its two datasets'
point counts / coord ranges / confirmed class overlap are filled in -- no silent assumptions
about shapes you haven't actually looked at. This mirrors the Stage-1 review's core lesson:
check what the data and checkpoint actually are before trusting a result built on them.

Conditions (3 seeds each, 9 runs total):
  1. ModelNet40 baseline:          random-init encoder, train+test on ModelNet40 (in-domain)
  2. Sim-to-real gap, no pretrain: random-init encoder, train on ModelNet40, test on ScanObjectNN
  3. Sim-to-real WITH pretrain:    ShapeNet-pretrained encoder (from pretrain_v2.py), fine-tuned
                                   on ModelNet40, tested on ScanObjectNN

Primary result: accuracy(condition 3) - accuracy(condition 2), on the shared class subset only
(ModelNet40 and ScanObjectNN do NOT share all classes or label indices -- evaluating on the
full label sets would silently compare apples to oranges).

Two pretrained checkpoints exist from Stage 1 -- pass --pretrained explicitly, don't default to
one, since they differ a lot (s0_rotz/best.pth: rotation-only; corrupt_aug/best.pth: corruption
augmented, better occ/clutter/scan but from an earlier, more brittle checkpoint). Recommend
running condition 3 with EACH checkpoint rather than picking one -- it's the more interesting
comparison and costs 2x3=6 runs instead of 3, cheap relative to Stage 1 training time.

Usage:
  python3 inspect_data.py --modelnet40 <path> --scanobjectnn <path>   # ALWAYS FIRST
  python3 finetune_stage2.py --report dataset_report.json \
      --modelnet40 <path> --scanobjectnn <path> \
      --pretrained /path/to/stage1_v2/runs/s0_rotz/best.pth \
      --pretrained /path/to/stage1_v2/runs/corrupt_aug/best.pth \
      --seeds 0 1 2 --work-dir stage2_results

Loads the Stage 1 encoder architecture directly from pretrain_v2.py (must be importable,
same directory or on PYTHONPATH) so condition 3 uses the EXACT trained weights/architecture,
not a re-implementation that could silently drift from what was actually pretrained.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pretrain_v2 import Encoder, load_encoder, n_params
except ImportError:
    sys.exit("Could not import pretrain_v2.py -- put finetune_stage2.py in the same directory, "
             "or add that directory to PYTHONPATH. Stage 2 must use the exact Stage 1 encoder "
             "class, not a re-implementation.")


def log(msg, path=None):
    print(msg, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(msg + "\n")


# =============================================================================
# Dataset loading -- gated on inspect_data.py's report, not assumptions
# =============================================================================

STOP_NOT_READY = 2  # distinct from argparse's exit(2) on bad args -- both must not read as success


def require_report(report_path):
    if not os.path.exists(report_path):
        sys.exit(f"{report_path} not found. Run inspect_data.py first -- this script will not "
                 f"guess your data's point count, coordinate range, or class overlap.")
    report = json.loads(Path(report_path).read_text())
    for key in ("modelnet40", "scanobjectnn"):
        entries = report.get(key)
        if not entries or not isinstance(entries, list):
            sys.exit(f"dataset_report.json has no entries under '{key}'. Re-run inspect_data.py "
                     f"with --{key} pointed at the real file.")
        e = entries[0]
        missing = [k for k in ("data_shape", "label_values", "coord_min", "coord_max") if k not in e]
        if missing:
            sys.exit(f"dataset_report.json's '{key}' entry is missing {missing} -- this report was "
                     f"not produced by a successful inspect_data.py run on a real file (presence of "
                     f"the key alone is not enough; a report with an empty dict for this dataset "
                     f"would otherwise pass this gate). Re-run inspect_data.py.")
    return report


def load_h5_dataset(path, n_points, data_key=None, label_key=None):
    import h5py
    with h5py.File(path, "r") as f:
        data_key = data_key or next(k for k in f.keys() if k.lower() in ("data", "points", "point_cloud"))
        label_key = label_key or next(k for k in f.keys() if "label" in k.lower())
        pts = f[data_key][:].astype(np.float32)
        labels = f[label_key][:].reshape(-1).astype(np.int64)
    # Normalize to ShapeNet's unit-sphere convention (pretrain_v2.py trained on this).
    pts = pts - pts.mean(1, keepdims=True)
    pts = pts / np.linalg.norm(pts, axis=2).max(1, keepdims=True).clip(min=1e-6)[:, :, None]
    # Random subsample to n_points (fast, NOT farthest-point -- this stage needs many runs;
    # pretrain_v2.py's FPS caching is for Stage 1 only, see its docstring).
    if pts.shape[1] != n_points:
        rng = np.random.default_rng(0)
        if pts.shape[1] > n_points:
            idx = np.stack([rng.choice(pts.shape[1], n_points, replace=False) for _ in range(len(pts))])
        else:
            idx = np.stack([rng.choice(pts.shape[1], n_points, replace=True) for _ in range(len(pts))])
        pts = np.take_along_axis(pts, idx[:, :, None], axis=1)
    return pts, labels


def restrict_to_shared_classes(pts, labels, name_of_label, shared_names):
    """name_of_label: dict mapping this dataset's integer label -> class NAME string.
    Filters to only the classes in shared_names, remaps to a common 0..K-1 index."""
    shared_sorted = sorted(shared_names)
    remap = {name: i for i, name in enumerate(shared_sorted)}
    keep_mask = np.array([name_of_label.get(int(l)) in remap for l in labels])
    new_labels = np.array([remap[name_of_label[int(l)]] for l in labels[keep_mask]], dtype=np.int64)
    return pts[keep_mask], new_labels, shared_sorted


# =============================================================================
# Classifier head + training loop (shared across all 3 conditions)
# =============================================================================

class ClsHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
                                 nn.Linear(256, n_classes))

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


EVAL_BATCH = 32  # same order as the training batch -- whole-tensor eval OOMs on real dataset sizes
# (confirmed during testing: unbatched eval of a ~70-shape fake set already needed batch=4 to avoid
# an OOM-kill on a 4GB sandbox; the reviewer's independent measurement put real peak memory growth
# at ~30 MiB/object at 1024 points, i.e. multiple GiB at a few hundred held-out objects)


@torch.no_grad()
def encode_batched(enc, X, batch_size=EVAL_BATCH, gen=None):
    """Run eval/inference through the encoder in chunks, not as one giant tensor."""
    enc.eval()
    out = []
    for i in range(0, len(X), batch_size):
        out.append(enc(X[i:i + batch_size]))
    return torch.cat(out)


def eval_rng(dev, seed):
    """Isolated generator for every eval-time stochastic op (the encoder's FPS included), so
    repeated evaluation doesn't consume/perturb the training RNG stream and is itself repeatable
    call to call. torch.random.fork_rng is the cleanest way to do this without touching
    pretrain_v2.py's Encoder.forward signature (it already accepts an optional `gen`)."""
    g = torch.Generator(device=dev if dev.type == "cuda" else "cpu")
    g.manual_seed(seed)
    return g


def train_one(enc, head, Xtr, ytr, Xva, yva, dev, epochs=100, lr=1e-3, finetune_encoder=True,
              log_path=None, eval_seed=0):
    params = list(head.parameters()) + (list(enc.parameters()) if finetune_encoder else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_va, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        enc.train(finetune_encoder), head.train()
        perm = torch.randperm(len(Xtr), device=dev)
        tot_loss = 0.0
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            with torch.set_grad_enabled(finetune_encoder):
                feat = enc(Xtr[idx])
            feat = feat if finetune_encoder else feat.detach()
            logits = head(feat)
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
        sched.step()
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            torch.manual_seed(eval_seed)  # eval uses its own fixed seed every epoch, isolated from
            head.eval()                   # the training stream (fork_rng restores the stream after)
            with torch.no_grad():
                feat = encode_batched(enc, Xva)
                va_acc = (head(feat).argmax(1) == yva).float().mean().item() * 100
        if va_acc > best_va:
            best_va = va_acc
            best_state = (
                {k: v.clone() for k, v in enc.state_dict().items()},
                {k: v.clone() for k, v in head.state_dict().items()},
            )
        if ep % 20 == 0 or ep == epochs:
            log(f"    ep {ep:3d}/{epochs} | train_loss {tot_loss/len(Xtr):.4f} | val_acc {va_acc:.1f} "
                f"(best {best_va:.1f})", log_path)
    enc.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    return best_va


def eval_acc(enc, head, X, y, dev, eval_seed=0):
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.manual_seed(eval_seed)
        head.eval()
        with torch.no_grad():
            feat = encode_batched(enc, X)
            pred = head(feat).argmax(1)
    oa = (pred == y).float().mean().item() * 100
    mca = float(np.mean([(pred[y == c] == c).float().mean().item() for c in y.unique() if (y == c).sum() > 0]))
    return oa, mca * 100


# =============================================================================
# Main experiment
# =============================================================================

def ckpt_identity(pretrained_path):
    """Full stem (parent dir + filename), not just the parent dir name -- 'run/best.pth' and
    'run/final.pth' previously collided on just Path(...).parent.name and silently overwrote
    each other's result.json / pooled into the same summary row."""
    if pretrained_path is None:
        return "none"
    p = Path(pretrained_path)
    return f"{p.parent.name}__{p.stem}"


def run_condition(cond, seed, mn_tr, mn_va, so_te, n_classes, pretrained_path, dev, work_dir, epochs,
                  finetune_encoder=True, select_metric="mca"):
    set_seed(seed)
    enc = Encoder().to(dev) if pretrained_path is None else load_encoder(pretrained_path, dev)
    head = ClsHead(enc.out_dim, n_classes).to(dev)
    ckpt_id = ckpt_identity(pretrained_path)
    run_dir = Path(work_dir) / f"{cond}_seed{seed}" / ckpt_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"
    log(f"=== condition={cond} seed={seed} pretrained={pretrained_path} finetune_encoder={finetune_encoder} "
        f"| encoder params {n_params(enc)/1e6:.2f}M", log_path)

    Xtr, ytr = mn_tr
    Xva, yva = mn_va
    best_val = train_one(enc, head, Xtr, ytr, Xva, yva, dev, epochs=epochs, log_path=log_path,
                         finetune_encoder=finetune_encoder, eval_seed=seed)

    # NOTE: mn_va is reused below as the ModelNet40 evaluation set -- there is currently no
    # separate held-out ModelNet40 test file plumbed in, so this is validation-set reuse, not an
    # untouched test estimate (train_one already selected the best epoch against this same set).
    # Labeled explicitly as "modelnet40_val_reused_as_eval" rather than "...test" so a reader
    # of result.json cannot mistake it for a clean test number. Replace with a real held-out
    # ModelNet40 test split before reporting this as a benchmark result.
    results = {"condition": cond, "seed": seed, "pretrained": pretrained_path, "checkpoint_id": ckpt_id,
              "finetune_encoder": finetune_encoder, "select_metric": select_metric, "best_val_acc": best_val}
    oa, mca = eval_acc(enc, head, Xva, yva, dev, eval_seed=seed)
    results["modelnet40_val_reused_as_eval"] = {"oa": oa, "mca": mca}
    log(f"  ModelNet40 (val, reused -- NOT a held-out test): oa {oa:.1f} mca {mca:.1f}", log_path)
    if so_te is not None:
        oa, mca = eval_acc(enc, head, *so_te, dev, eval_seed=seed)
        results["scanobjectnn_test"] = {"oa": oa, "mca": mca}
        log(f"  ScanObjectNN test (sim-to-real): oa {oa:.1f} mca {mca:.1f}", log_path)

    (run_dir / "result.json").write_text(json.dumps(results, indent=1))
    # Save the actual selected weights, not just aggregate metrics -- previously only the
    # in-memory best state existed; a crash/restart lost it even though the sweep continued.
    torch.save({"encoder": enc.state_dict(), "encoder_cfg": enc.cfg, "head": head.state_dict()},
              run_dir / "selected_state.pth")
    return results



SO_NAMES = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table",
            "bed", "pillow", "sink", "sofa", "toilet"]  # order confirmed via label-count histogram


def load_dataset_dir(path, n_points, pattern):
    """path = one .h5 file, or a directory: load + concatenate every file matching pattern."""
    import glob, os
    files = sorted(glob.glob(os.path.join(path, pattern))) if os.path.isdir(path) else [path]
    assert files, f"no files matching {pattern} under {path}"
    parts = [load_h5_dataset(f, n_points) for f in files]
    print(f"loaded {len(files)} file(s) from {path}: {sum(len(q[1]) for q in parts)} shapes")
    return np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="dataset_report.json")
    ap.add_argument("--modelnet40", required=True)
    ap.add_argument("--scanobjectnn", required=True)
    ap.add_argument("--pretrained", action="append", default=[],
                    help="Path to a Stage 1 best.pth. Repeat the flag to compare multiple checkpoints "
                         "(recommended: pass both s0_rotz/best.pth and corrupt_aug/best.pth).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--work-dir", default="stage2_results")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="Linear-probe comparison (frozen pretrained features) instead of full "
                         "fine-tuning. The proposal describes both; default is full fine-tuning, "
                         "matching condition 3's wording, but run both if the frozen comparison "
                         "from the broader method description also matters for your writeup.")
    args = ap.parse_args()

    report = require_report(args.report)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    # --- Load raw data. NOTE: class-name mapping from integer label requires the real label
    # metadata (a classes.txt / names list shipped with the dataset), NOT the guessed overlap
    # inspect_data.py prints -- that guess is only for a sanity check before you get here.
    mn_rep, so_rep = report["modelnet40"][0], report["scanobjectnn"][0]
    print(f"ModelNet40 report says: {mn_rep.get('data_shape')}, {len(mn_rep.get('label_values', []))} classes")
    print(f"ScanObjectNN report says: {so_rep.get('data_shape')}, {len(so_rep.get('label_values', []))} classes")
    print("!! This script does not yet have the real label->name mapping for either dataset wired "
          "in -- that needs whatever classes.txt / shape_names.txt ships alongside the .h5 files. "
          "Fill in `name_of_label` for each dataset below before this produces a trustworthy "
          "shared-class result. Until then this is a scaffold, not a run-it-and-trust-it script.")

    mn_pts, mn_labels = load_dataset_dir(args.modelnet40, args.n_points, "ply_data_train*.h5")
    so_pts, so_labels = load_dataset_dir(args.scanobjectnn, args.n_points, "*_objectdataset.h5")

    # PLACEHOLDER mapping -- replace with the real shape_names.txt / classes.txt content.
    # Left as an identity-ish stand-in so the rest of the pipeline (splits, training loop,
    # metrics) can be exercised and reviewed before the real label files are wired in.
    import os as _os
    mn_dir = args.modelnet40 if _os.path.isdir(args.modelnet40) else _os.path.dirname(args.modelnet40)
    with open(_os.path.join(mn_dir, "shape_names.txt")) as f:
        mn_names = {i: ln.strip() for i, ln in enumerate(f) if ln.strip()}
    so_names = dict(enumerate(SO_NAMES))
    assert len(mn_names) == 40 and int(mn_labels.max()) == 39, "ModelNet40 label/name mismatch"
    assert len(so_names) == 15 and int(so_labels.max()) == 14, "ScanObjectNN label/name mismatch"
    shared = set(mn_names.values()) & set(so_names.values())  # empty with placeholders -- expected

    if not shared:
        print("\n[STOP] No shared classes resolved (expected with the placeholder name mapping above). "
              "Wire in the real class-name files for both datasets, then re-run. Not proceeding to "
              "training with an empty/undefined class set.")
        sys.exit(STOP_NOT_READY)  # was: bare `return` -> exit code 0, indistinguishable from "ran fine"

    mn_pts_s, mn_labels_s, class_names = restrict_to_shared_classes(mn_pts, mn_labels, mn_names, shared)
    so_pts_s, so_labels_s, _ = restrict_to_shared_classes(so_pts, so_labels, so_names, shared)
    n_classes = len(class_names)
    print(f"shared classes ({n_classes}): {class_names}")

    def to_t(x):
        return torch.from_numpy(x).to(dev)

    rng = np.random.default_rng(0)
    n = len(mn_pts_s)
    idx = rng.permutation(n)
    n_va = int(n * args.val_frac)
    va_idx, tr_idx = idx[:n_va], idx[n_va:]
    mn_tr = (to_t(mn_pts_s[tr_idx]), to_t(mn_labels_s[tr_idx]))
    mn_va = (to_t(mn_pts_s[va_idx]), to_t(mn_labels_s[va_idx]))
    so_te = (to_t(so_pts_s), to_t(so_labels_s))

    # Conditions 1 ("ModelNet40 baseline") and 2 ("sim2real, no pretrain") previously trained an
    # identical from-scratch model twice per seed and just evaluated it on different domains.
    # Train once, evaluate on both -- 9 fits total (3 scratch + 2 checkpoints x 3 seeds) instead
    # of 12, with no change to what each condition measures.
    all_results = []
    for seed in args.seeds:
        r = run_condition("c1c2_scratch", seed, mn_tr, mn_va, so_te, n_classes, None, dev,
                          args.work_dir, args.epochs, finetune_encoder=not args.freeze_encoder)
        r_c1 = {**r, "condition": "c1_modelnet_baseline"}
        r_c2 = {**r, "condition": "c2_sim2real_no_pretrain"}
        all_results += [r_c1, r_c2]
        for ckpt in args.pretrained:
            all_results.append(run_condition("c3_sim2real_pretrained", seed, mn_tr, mn_va, so_te,
                                             n_classes, ckpt, dev, args.work_dir, args.epochs,
                                             finetune_encoder=not args.freeze_encoder))

    Path(args.work_dir, "all_results.json").write_text(json.dumps(all_results, indent=1))

    def headline_key(cond):
        # c1's headline is the ModelNet40 (val-reused) number; everything downstream of
        # sim-to-real (c2, c3) headlines on the ScanObjectNN number -- this must match whatever
        # metric train_one actually selected the checkpoint on (currently val OA, see train_one;
        # headline here is mCA -- document and justify this split per the review, don't silently
        # select on one metric and report another without saying so).
        return "modelnet40_val_reused_as_eval" if cond == "c1_modelnet_baseline" else "scanobjectnn_test"

    def mean_std(cond, ckpt_id=None):
        rows = [r for r in all_results if r["condition"] == cond and
                (ckpt_id is None or r.get("checkpoint_id") == ckpt_id)]
        if not rows:
            return None
        key = headline_key(cond)
        accs = [r[key]["mca"] for r in rows if key in r]
        if not accs:
            return None
        return np.mean(accs), np.std(accs), len(accs)

    print("\n=== Stage 2 summary (mean class accuracy %, mean +/- std across seeds) ===")
    print("NOTE: checkpoint selection used val OA (train_one); this table reports mCA -- a "
          "deliberate metric split per the review's point #, not an oversight, but the two can "
          "disagree even for the identical selected state.")
    for cond in ["c1_modelnet_baseline", "c2_sim2real_no_pretrain"]:
        m = mean_std(cond)
        if m:
            print(f"{cond:30s} {m[0]:5.1f} +/- {m[1]:4.1f}  (n={m[2]})")
    for ckpt in args.pretrained:
        cid = ckpt_identity(ckpt)
        m = mean_std("c3_sim2real_pretrained", cid)
        if m:
            print(f"c3_sim2real_pretrained[{cid:28s}] {m[0]:5.1f} +/- {m[1]:4.1f}  (n={m[2]})")
    gap_base = mean_std("c2_sim2real_no_pretrain")
    if gap_base:
        print(f"\nPrimary result: condition-3 mca minus condition-2 mca ({gap_base[0]:.1f}), per checkpoint:")
        for ckpt in args.pretrained:
            cid = ckpt_identity(ckpt)
            m3 = mean_std("c3_sim2real_pretrained", cid)
            if m3:
                print(f"  {cid}: {m3[0] - gap_base[0]:+.1f} points")


if __name__ == "__main__":
    main()
