"""Validate the NoW harness before trusting any number it produces.

Two baselines, neither of which needs a trained encoder or DECA:

  identity  Feed each ground-truth scan back as its own prediction, with the
            ground-truth landmarks. A correct metric must return ~0. This
            exercises mesh loading, landmark alignment and scan-to-mesh distance
            end to end, and is the single strongest check available offline.

  mean      Predict FLAME's mean face for every image, using our own 7
            landmarks. Should return a large but stable number. This is the
            check that our landmark emitter and prediction layout are right,
            because the identity baseline never touches either.

Run `identity` first: if it is not ~0, nothing downstream means anything.
"""

import sys, pathlib, shutil, argparse
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d import assets, now
from face3d.flame_torch import FlameTorch
from face3d.landmarks import LandmarkEmbedding

ROOT = pathlib.Path(__file__).resolve().parents[1]


def build(mode, pred_root, n_subjects):
    pred_root = pathlib.Path(pred_root)
    if pred_root.exists():
        shutil.rmtree(pred_root)
    pred_root.mkdir(parents=True)

    images = now.image_list(ROOT, "validation")
    # One image per subject keeps the run short; the metric skips any image in
    # the list without a matching prediction, so a subset list is required.
    seen, subset = set(), []
    for rel in images:
        s = rel.split("/")[0]
        if s not in seen:
            seen.add(s); subset.append(rel)
        if len(seen) >= n_subjects:
            break

    if mode == "mean":
        flame = FlameTorch(assets.model_path_or_skip())
        emb = LandmarkEmbedding(ROOT / "mediapipe_landmark_embedding" /
                                "mediapipe_landmark_embedding.npz")
        verts, _ = flame(batch_size=1)
        lmk = now.landmarks_7(emb, verts, flame.faces)[0].numpy()
        v, f = verts[0].numpy(), flame.faces.numpy()
        for rel in subset:
            now.write_prediction(pred_root, rel, v, f, lmk)
    else:
        for rel in subset:
            subj = rel.split("/")[0]
            scan = next((ROOT / "scans" / subj).glob("*.obj"))
            pp = next((ROOT / "scans_lmks_onlypp" / subj).glob("*.pp"))
            parts = pathlib.Path(rel).parts
            d = pred_root / parts[0] / parts[1]
            d.mkdir(parents=True, exist_ok=True)
            stem = pathlib.Path(parts[2]).stem
            shutil.copyfile(scan, d / f"{stem}.obj")
            np.save(d / f"{stem}.npy", now.load_pp(pp).astype(np.float32))

    listing = pred_root / "subset.txt"
    listing.write_text("\n".join(subset) + "\n")
    return subset, listing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["identity", "mean"])
    ap.add_argument("--subjects", type=int, default=5)
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()

    pred_root = ROOT / "out" / f"now_{a.mode}"
    subset, listing = build(a.mode, pred_root, a.subjects)
    print(f"=== NoW harness validation: {a.mode} baseline ===")
    print(f"    {len(subset)} predictions in {pred_root.relative_to(ROOT)}")
    if a.mode == "identity":
        print("    expectation: error ~0 mm (prediction IS the ground truth)")
    else:
        print("    expectation: a large but stable number (FLAME mean face)")

    rc, out = now.run_docker_eval(ROOT, pred_root, nproc=a.nproc, imgs_list=listing)
    for line in out.splitlines():
        if any(k in line for k in ("missing", "computed distances", "Error", "Traceback")):
            print("   " + line.strip())

    # The container writes the headline numbers to a sidecar file, not stdout.
    res = sorted(pred_root.glob("results/*.meanmedian"))
    if res:
        # compute_error.py writes median on line 1, mean on line 2 -- not the
        # order the ".meanmedian" filename suggests.
        median, mean = (float(x) for x in res[0].read_text().split())
        print("")
        print(f"   median {median:.6g} mm")
        print(f"   mean   {mean:.6g} mm")
        if a.mode == "identity":
            ok = max(median, mean) < 1e-3
            verdict = "PASS - metric pipeline is correct" if ok else "FAIL - not ~0"
            print("")
            print(f"   {verdict}")
    else:
        print("   no results file written")
    print("")
    print(f"[docker exit {rc}]")
