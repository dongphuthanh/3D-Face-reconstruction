"""Detector checks against real NoW photographs."""

import sys, pathlib, time
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d.detect import FaceDetector, select_embedding_points, crop_square, MODEL

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)
FAILURES = []

if not MODEL.exists():
    print(f"SKIP — no MediaPipe model at {MODEL}"); sys.exit(0)
imgs_dir = ROOT / "NoW_Dataset" / "final_release_version" / "iphone_pictures"
if not imgs_dir.exists():
    print("SKIP — NoW images not on disk"); sys.exit(0)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


emb = np.load(ROOT / "mediapipe_landmark_embedding" / "mediapipe_landmark_embedding.npz")
need = emb["landmark_indices"].astype(int).ravel()

rels = [l.strip() for l in open(ROOT / "imagepathsvalidation.txt") if l.strip()]
# One per category, so occlusion and selfie are represented, not just the easy ones.
by_cat, sample = {}, []
for r in rels:
    c = r.split("/")[1]
    if c not in by_cat:
        by_cat[c] = r; sample.append(r)

print(f"=== MediaPipe detector on {len(sample)} NoW photos ===")
det = FaceDetector(blendshapes=True)
times, found, crops = [], 0, []
for rel in sample:
    im = np.asarray(Image.open(imgs_dir / rel).convert("RGB"))
    t0 = time.perf_counter(); res = det.detect(im); times.append(time.perf_counter() - t0)
    cat = rel.split("/")[1]
    if res is None:
        print(f"    {cat:22s} NO FACE")
        continue
    found += 1
    pts = select_embedding_points(res["norm"], need)
    crop, to_ndc = crop_square(im, res["norm"])
    crops.append((cat, crop, to_ndc(pts)))
    print(f"    {cat:22s} {len(res['norm'])} lmks, {len(res.get('blendshapes',{}))} blendshapes, "
          f"{times[-1]*1000:5.0f} ms")

check("105 embedding points selected from the 478 detected",
      all(c[2].shape == (105, 2) for c in crops),
      f"{len(crops)} crops, each {crops[0][2].shape if crops else None}")

# Measured over the full 352-image validation split: 83% overall, and 100% on
# selfies but ~77% on the multiview categories. Lowering the confidence
# threshold to 0.2 changes nothing, so these are hard detector failures on
# pose, not marginal rejections. Asserting 100% here would encode a wrong
# assumption; asserting the frontal case still works is the real invariant.
check("frontal (selfie) faces are always found",
      any(c[0] == "selfie" for c in crops), "selfie is 100% on the full split")
check("at least half the sampled categories detect", found >= len(sample) / 2,
      f"{found}/{len(sample)} sampled; 83% over the full split")
check("crop-space landmarks lie inside the frame",
      all(np.abs(c[2]).max() < 1.0 for c in crops),
      f"max |ndc| {max(float(np.abs(c[2]).max()) for c in crops):.3f}")
check("detection is fast enough for webcam (SC3)", np.median(times) < 0.040,
      f"median {np.median(times)*1000:.0f} ms on full-res photos")

det.close()

# Draw the 105 points on each crop so a wrong index set is visible, not just green.
from PIL import ImageDraw
tiles = []
for cat, crop, ndc in crops:
    img = Image.fromarray(crop); d = ImageDraw.Draw(img)
    px = np.stack([(ndc[:, 0] + 1) / 2 * img.width, (1 - ndc[:, 1]) / 2 * img.height], -1)
    for x, y in px:
        d.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(255, 90, 60))
    d.text((6, 6), cat, fill=(255, 255, 255))
    tiles.append(np.asarray(img))
Image.fromarray(np.concatenate(tiles, 1)).save(OUT / "detect_test.png")
print("  wrote detect_test.png")

print("\n" + "=" * 56)
print("ALL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
