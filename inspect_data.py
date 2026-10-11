#!/usr/bin/env python3
"""
Stage 2, step 1 (run this FIRST, before finetune_stage2.py exists in final form).

Inspects whatever ModelNet40 / ScanObjectNN files you actually have on the server and
prints everything finetune_stage2.py needs to know before it can be trusted:
  * file format (HDF5 vs raw .txt/.off) and the actual key names inside
  * point count per shape as shipped (often 2048, NOT the 1024 pretrain_v2.py used)
  * coordinate range (is it already unit-sphere normalized like ShapeNet, or not?)
  * class list + count per class, for BOTH datasets
  * the overlapping class subset between the two label sets (by NAME, not index --
    ModelNet40 and ScanObjectNN do not share label-index conventions)

Never assume any of this. finetune_stage2.py reads a JSON this script writes
(dataset_report.json) and refuses to run if the two datasets' assumptions don't match it.

Usage:
  python3 inspect_data.py --modelnet40 /path/to/modelnet40.h5 --scanobjectnn /path/to/scanobjectnn.h5
  python3 inspect_data.py --modelnet40-dir /path/to/dir  (auto-finds *.h5 / *.off / *.txt inside)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

# Canonical ScanObjectNN class names (the 15-class OBJ_ONLY / OBJ_BG split used in most papers).
# ModelNet40's 40 names are the standard list. We match by NAME after lowercasing + singularizing
# common plurals, since the two datasets don't share a label-index convention.
MODELNET40_NAMES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard",
    "lamp", "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio",
    "range_hood", "sink", "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]
SCANOBJECTNN_NAMES = [
    "bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table",
    "bed", "pillow", "sink", "sofa", "toilet",
]


def norm_name(s):
    return s.strip().lower().replace(" ", "_").rstrip("s")


def find_h5(path_or_dir):
    if os.path.isfile(path_or_dir):
        return [path_or_dir]
    return sorted(glob.glob(os.path.join(path_or_dir, "**", "*.h5"), recursive=True))


def inspect_h5(path):
    import h5py
    report = {"path": path, "format": "hdf5"}
    with h5py.File(path, "r") as f:
        report["keys"] = list(f.keys())
        data_key = next((k for k in f.keys() if k.lower() in ("data", "points", "point_cloud")), None)
        label_key = next((k for k in f.keys() if "label" in k.lower()), None)
        report["data_key"], report["label_key"] = data_key, label_key
        if data_key:
            d = f[data_key]
            report["data_shape"] = list(d.shape)
            sample = d[: min(200, d.shape[0])]
            report["coord_min"] = sample.min(axis=(0, 1)).tolist()
            report["coord_max"] = sample.max(axis=(0, 1)).tolist()
            r = np.linalg.norm(sample - sample.mean(1, keepdims=True), axis=2).max()
            report["approx_max_radius_after_centering"] = float(r)
        if label_key:
            labels = f[label_key][:]
            report["label_shape"] = list(labels.shape)
            vals, counts = np.unique(labels, return_counts=True)
            report["label_values"] = vals.tolist()
            report["count_per_label"] = dict(zip(vals.astype(str).tolist(), counts.tolist()))
    return report


def guess_overlap(mn_names, so_names):
    mn_norm = {norm_name(n): n for n in mn_names}
    so_norm = {norm_name(n): n for n in so_names}
    shared = sorted(set(mn_norm) & set(so_norm))
    return [{"shared_name": s, "modelnet40_name": mn_norm[s], "scanobjectnn_name": so_norm[s]} for s in shared]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelnet40")
    ap.add_argument("--modelnet40-dir")
    ap.add_argument("--scanobjectnn")
    ap.add_argument("--scanobjectnn-dir")
    ap.add_argument("--out", default="dataset_report.json")
    args = ap.parse_args()

    report = {"modelnet40": [], "scanobjectnn": []}
    for key, single, d in [("modelnet40", args.modelnet40, args.modelnet40_dir),
                           ("scanobjectnn", args.scanobjectnn, args.scanobjectnn_dir)]:
        target = single or d
        if not target:
            print(f"[warn] no --{key} or --{key}-dir given, skipping")
            continue
        files = find_h5(target) if os.path.isdir(target) else [target]
        if not files:
            print(f"[warn] no .h5 files found under {target} -- if this dataset ships as "
                  f".off/.txt per-shape files instead, this script needs a different branch; "
                  f"tell me the actual file layout (ls -R a few levels) before writing finetune code")
            continue
        for fp in files:
            try:
                rep = inspect_h5(fp)
                report[key].append(rep)
                print(f"\n=== {key}: {fp} ===")
                print(f"  keys: {rep['keys']}")
                print(f"  data[{rep.get('data_key')}] shape: {rep.get('data_shape')}")
                print(f"  coord range: min {rep.get('coord_min')}  max {rep.get('coord_max')}")
                print(f"  max radius after centering (sample of 200): {rep.get('approx_max_radius_after_centering'):.3f}"
                      if rep.get("approx_max_radius_after_centering") is not None else "")
                print(f"  label[{rep.get('label_key')}] shape: {rep.get('label_shape')} "
                      f"| classes: {len(rep.get('label_values', []))}")
            except Exception as e:  # noqa: BLE001
                print(f"[FAILED] {fp}: {type(e).__name__}: {e}")

    report["overlap_by_name_guess"] = guess_overlap(MODELNET40_NAMES, SCANOBJECTNN_NAMES)
    print(f"\n=== guessed class overlap (by NAME, verify against the real label files -- "
          f"this is a guess from the standard published class lists, not read from your data) ===")
    for o in report["overlap_by_name_guess"]:
        print(f"  {o['shared_name']}")
    print(f"  -> {len(report['overlap_by_name_guess'])} shared classes")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nWrote {args.out}. finetune_stage2.py will refuse to run without a report that has "
          f"BOTH datasets' point counts, coordinate ranges, and the real (not guessed) class overlap "
          f"confirmed against actual label metadata, not the standard-list guess above.")


if __name__ == "__main__":
    main()
