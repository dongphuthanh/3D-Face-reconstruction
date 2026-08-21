"""Run a trained encoder over the NoW split and write predictions for scoring.

Cropping strategy. Training crops came from MediaPipe's detected landmarks, so
inference should match. But MediaPipe fails on 17% of NoW (pose, not occlusion),
and NoW requires a prediction for every image.

The resolution is that the *encoder needs no landmarks at inference* -- it maps
pixels to coefficients, and landmarks only ever existed to supply a training
loss. NoW ships a bounding box for every image, so undetected faces are cropped
from that box instead and encoded normally. No mean-face fallback is required.

The bbox path is still a distribution shift relative to training, so predictions
are tagged with which path produced them and the two are reported separately.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d import assets, now
from face3d.detect import FaceDetector, crop_square
from face3d.encoder import ResNetEncoder
from face3d.flame_torch import FlameTorch
from face3d.landmarks import LandmarkEmbedding
from face3d.params import FlameParams
from face3d.pipeline import FaceRenderer

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES = ROOT / "NoW_Dataset" / "final_release_version" / "iphone_pictures"
BOXES = ROOT / "NoW_Dataset" / "final_release_version" / "detected_face"


def bbox_crop(image, rel, size=224, margin=1.25):
    """Square crop from NoW's supplied bounding box, padded to match training.

    NoW's boxes are tight to the face; the MediaPipe crops used in training carry
    a 1.6x margin around the landmark extent. Squaring the box and expanding it
    approximates that framing.
    """
    p = (BOXES / rel).with_suffix(".npy")
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True, encoding="latin1").item()
    top, left = float(d["top"]), float(d["left"])
    bottom, right = float(d["bottom"]), float(d["right"])
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half = max(right - left, bottom - top) * margin / 2
    box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    return np.asarray(Image.fromarray(image).crop(box).resize((size, size), Image.BILINEAR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs" / "ffhq" / "encoder.pt"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default=str(ROOT / "out" / "now_pred"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--crop-margin", type=float, default=1.6,
                    help="must match the margin the model was ingested with. "
                         "Arc2Face used 1.15 (its sources are pre-cropped); FFHQ "
                         "and DigiFace used 1.6. A mismatch is a distribution "
                         "shift that shows up as inflated deviation and a "
                         "collapsed identity ratio")
    ap.add_argument("--shape-scale", type=float, default=None,
                    help="override the encoder's calibration; defaults to "
                         "SHAPE_CALIBRATION. Set 1.0 for the raw prediction")
    ap.add_argument("--neutral", action="store_true",
                    help="zero expression and pose before writing the mesh")
    a = ap.parse_args()

    ckpt = pathlib.Path(a.checkpoint)
    if not ckpt.exists():
        print(f"SKIP - no checkpoint at {ckpt}; run scripts/train.py first")
        sys.exit(0)

    flame = FlameTorch(assets.model_path_or_skip()).to(DEV)
    emb = LandmarkEmbedding(
        ROOT / "mediapipe_landmark_embedding" / "mediapipe_landmark_embedding.npz",
        device=DEV)
    enc = ResNetEncoder(n_shape=100, n_expr=50, pretrained=False).to(DEV)
    enc.load_state_dict(torch.load(ckpt, map_location=DEV)["model"])
    enc.eval()

    rels = now.image_list(ROOT, a.split)
    if a.limit:
        rels = rels[: a.limit]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    det = FaceDetector(blendshapes=False)
    faces_np = flame.faces.cpu().numpy()
    stats = {"detected": 0, "bbox": 0, "failed": 0}
    path_of = {}
    t0 = time.time()

    for i, rel in enumerate(rels, 1):
        src = IMAGES / rel
        if not src.exists():
            stats["failed"] += 1
            continue
        im = np.asarray(Image.open(src).convert("RGB"))

        res = det.detect(im)
        if res is not None:
            crop, _ = crop_square(im, res["norm"], size=a.size, margin=a.crop_margin)
            stats["detected"] += 1
            path_of[rel] = "mediapipe"
        else:
            crop = bbox_crop(im, rel, size=a.size)
            if crop is None:
                stats["failed"] += 1
                continue
            stats["bbox"] += 1
            path_of[rel] = "bbox"

        x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)[None]
        x = x.float().to(DEV) / 255.0
        with torch.no_grad():
            pred = enc.predict(x)
            from face3d.encoder import SHAPE_CALIBRATION
            scale = SHAPE_CALIBRATION if a.shape_scale is None else a.shape_scale
            if scale != 1.0:
                pred.shape.mul_(scale)
            if a.neutral:
                # Measured, not assumed: NoW's ground truth is one neutral scan
                # per subject, and zeroing expression and pose improves the
                # median from 1.983 mm to 1.766 mm. Use --neutral for any
                # reported number.
                pred.expr.zero_()
                pred.pose.zero_()
            verts, _ = FaceRenderer(flame, image_size=a.size).geometry(pred)
            lmk7 = now.landmarks_7(emb, verts, flame.faces)[0].cpu().numpy()

        now.write_prediction(out, rel, verts[0].cpu().numpy(), faces_np, lmk7)
        if i % 50 == 0 or i == len(rels):
            print(f"  {i:4d}/{len(rels)}  detected={stats['detected']} "
                  f"bbox={stats['bbox']} failed={stats['failed']}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    det.close()
    (out / "predict_stats.json").write_text(json.dumps(stats, indent=1))
    (out / "crop_path.json").write_text(json.dumps(path_of, indent=1))
    n = stats["detected"] + stats["bbox"]
    print("")
    print(f"  wrote {n} predictions to {out}")
    print(f"  via mediapipe crop: {stats['detected']}  via NoW bbox: {stats['bbox']}  "
          f"failed: {stats['failed']}")


if __name__ == "__main__":
    main()
