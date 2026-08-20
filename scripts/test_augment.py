"""Verify augmented landmarks match what a detector independently finds.

The failure mode this guards against is silent: apply an affine to the pixels,
forget to apply it to the targets, and the landmark loss trains against wrong 2D
positions. Nothing crashes, the loss still falls, and the model is simply worse
for no visible reason.

So the check does not compare my transform against my own maths. It augments the
image, runs MediaPipe on the augmented pixels, and compares the detector's
answer to the transformed landmarks.
"""

import pathlib
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d.augment import (affine_view, consistency_loss, photometric_jitter,
                            sample_params, two_views)
from face3d.detect import MODEL, FaceDetector, select_embedding_points

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
FAILURES = []

crops = ROOT / "data" / "ffhq" / "crops"
cache = ROOT / "data" / "ffhq" / "landmarks_224.npz"
if not (MODEL.exists() and cache.exists()):
    print("SKIP - needs the MediaPipe model and an FFHQ ingest")
    sys.exit(0)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


z = np.load(cache)
keys, lmks = z["keys"], z["landmarks"]
emb = np.load(ROOT / "mediapipe_landmark_embedding" / "mediapipe_landmark_embedding.npz")
need = emb["landmark_indices"].astype(int).ravel()

N = 8
imgs = torch.stack([
    torch.from_numpy(np.array(Image.open(crops / f"{k}.jpg").convert("RGB"), np.uint8))
    .permute(2, 0, 1).float() / 255.0
    for k in keys[:N]])
lm = torch.from_numpy(lmks[:N])

print("=== geometric augmentation vs an independent detector ===")
torch.manual_seed(0)
p = sample_params(N, imgs.device)
aug_img, aug_lm = affine_view(imgs, lm, p)

det = FaceDetector(blendshapes=False)
errs, found = [], 0
for i in range(N):
    arr = (aug_img[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    res = det.detect(arr)
    if res is None:
        continue
    found += 1
    pts = select_embedding_points(res["norm"], need)
    ndc = np.stack([pts[:, 0] * 2 - 1, 1 - pts[:, 1] * 2], -1)
    errs.append(float(np.abs(ndc - aug_lm[i].numpy()).mean()))
det.close()

check("detector still finds the augmented faces", found >= N - 1, f"{found}/{N}")
mean_err = float(np.mean(errs)) if errs else 9.9
# Detector noise between an original and a resampled image is a few thousandths
# of NDC. A missing or inverted transform lands one to two orders higher.
check("transformed landmarks match the detector", mean_err < 0.03,
      f"mean |diff| {mean_err:.4f} NDC")

print("")
print("=== sanity of the transform itself ===")
ident = {"theta": torch.zeros(N), "scale": torch.ones(N),
         "tx": torch.zeros(N), "ty": torch.zeros(N)}
i_img, i_lm = affine_view(imgs, lm, ident)
check("identity transform leaves landmarks unchanged",
      float((i_lm - lm).abs().max()) < 1e-5)
check("identity transform leaves pixels unchanged",
      float((i_img - imgs).abs().max()) < 0.02,
      f"max |diff| {float((i_img - imgs).abs().max()):.4f} (resampling only)")

print("")
print("=== photometric jitter ===")
j = photometric_jitter(imgs)
check("jitter changes pixels", float((j - imgs).abs().mean()) > 0.01)
check("jitter stays in range", float(j.min()) >= 0 and float(j.max()) <= 1)

print("")
print("=== paired views ===")
vi, vl = two_views(imgs, lm)
check("two_views doubles the batch", vi.shape[0] == 2 * N and vl.shape[0] == 2 * N)
check("the two halves genuinely differ",
      float((vi[:N] - vi[N:]).abs().mean()) > 0.01)
same = consistency_loss(torch.cat([torch.ones(N, 100), torch.ones(N, 100)]))
diff = consistency_loss(torch.cat([torch.ones(N, 100), torch.zeros(N, 100)]))
check("consistency loss is zero when halves agree", float(same) == 0.0)
check("consistency loss is positive when they differ", float(diff) > 0, f"{float(diff):.1f}")

panel = torch.cat([torch.cat(list(imgs[:4].permute(0, 2, 3, 1)), 1),
                   torch.cat(list(vi[:4].permute(0, 2, 3, 1)), 1),
                   torch.cat(list(vi[N:N + 4].permute(0, 2, 3, 1)), 1)], 0)
Image.fromarray((panel.clamp(0, 1).numpy() * 255).astype(np.uint8)).save(OUT / "augment_test.png")
print("  wrote augment_test.png (original / view A / view B)")

print("")
print("=" * 56)
print("ALL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
