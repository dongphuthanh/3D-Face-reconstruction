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
from face3d.learn.augment import (affine_view, consistency_loss, photometric_jitter,
                            sample_params, scale_jitter, two_views)
from face3d.learn.detect import MODEL, FaceDetector, select_embedding_points

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

print("")
print("=== swap loss: collapse must not be a way out ===")
# The point of swapping rather than penalising distance. A distance penalty is
# minimised perfectly by a constant shape, which is what the measured run did:
# identity ratio fell 0.42 -> 0.29. Swapping gives collapse no reward, because
# the other view's shape then has to explain this view's face and a constant
# renders the mean.
from face3d.learn.augment import swap_shape
from face3d.geometry.params import FlameParams

B, NS = 4, 100
mk = lambda s: FlameParams(shape=s, expr=torch.zeros(2 * B, 50),
                           pose=torch.zeros(2 * B, 15), cam=torch.zeros(2 * B, 3),
                           light=torch.zeros(2 * B, 9, 3))
torch.manual_seed(0)
distinct = torch.randn(2 * B, NS)
p_swapped = swap_shape(mk(distinct))
check("swap exchanges the two halves",
      torch.equal(p_swapped.shape[:B], distinct[B:])
      and torch.equal(p_swapped.shape[B:], distinct[:B]))
check("swap leaves everything except shape untouched",
      torch.equal(p_swapped.expr, torch.zeros(2 * B, 50))
      and torch.equal(p_swapped.cam, torch.zeros(2 * B, 3)))

collapsed = torch.ones(2 * B, NS) * 0.3          # identical for every image
check("distance penalty rewards collapse (this is the bug)",
      float(consistency_loss(collapsed)) == 0.0
      and float(consistency_loss(distinct)) > 1.0,
      f"collapsed {float(consistency_loss(collapsed)):.1f} vs "
      f"distinct {float(consistency_loss(distinct)):.1f}")
check("swap is a no-op under collapse, so it grants no reward",
      torch.equal(swap_shape(mk(collapsed)).shape, collapsed),
      "swapped == original, so the reconstruction loss is unchanged and "
      "collapse buys nothing")

print("")
print("=== augmented views must not shift low-level statistics ===")
# BatchNorm tracks activation statistics over the training distribution and
# applies them to clean images at eval time. Zero-padded rotation corners shift
# those statistics enough that the encoder predicted a camera scale of 9.2
# instead of 6.9 purely from switching train() to eval() -- with the face then
# rendered off-frame. Nothing crashes; validation loss simply looks terrible
# while training loss looks fine.
black_before = float((imgs.sum(1) == 0).float().mean())
black_after = float((vi.sum(1) == 0).float().mean())
check("augmentation does not add large black regions",
      black_after - black_before < 0.02,
      f"exact-zero pixels {black_before * 100:.1f}% -> {black_after * 100:.1f}%")
check("mean brightness is preserved within jitter range",
      abs(float(vi.mean()) - float(imgs.mean())) < 0.12,
      f"{float(imgs.mean()):.3f} -> {float(vi.mean()):.3f}")
check("view A stays geometrically clean (anchors BatchNorm)",
      float((vl[:N] - lm).abs().max()) < 1e-5,
      "weak view leaves landmarks untouched")

# --- crop-scale jitter -------------------------------------------------------
# The only augmentation the identity-data path gets. It must move the landmarks
# with the pixels: applied to pixels alone it trains the encoder against
# silently wrong 2D targets, which reads as an accuracy problem rather than a
# bug. Checked against the detector and against the exact algebra.
print("")
print("=== crop-scale jitter ===")
S = 1.12
sj_img, sj_lm = scale_jitter(imgs, lm, S, S)
dev = float((sj_lm - lm * S).abs().max())
check("landmarks scale by exactly the zoom factor", dev < 1e-5, f"max deviation {dev:.2e}")

det2 = FaceDetector(blendshapes=False)
errs_sj, errs_raw = [], []
for i in range(N):
    arr = (sj_img[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    res = det2.detect(arr)
    if res is None:
        continue
    pts = select_embedding_points(res["norm"], need)
    ndc = np.stack([pts[:, 0] * 2 - 1, 1 - pts[:, 1] * 2], -1)
    errs_sj.append(float(np.abs(ndc - sj_lm[i].numpy()).mean()))
    errs_raw.append(float(np.abs(ndc - lm[i].numpy()).mean()))
det2.close()
err = float(np.mean(errs_sj)) if errs_sj else 9.9
base = float(np.mean(errs_raw)) if errs_raw else 0.0
check("zoomed image agrees with the transformed landmarks", err < base * 0.5,
      f"detector error {err:.4f}, against {base:.4f} for untransformed targets")

blk = float((sj_img.sum(1) == 0).float().mean())
check("zoom-out adds no black border (padding_mode=border)",
      blk - black_before < 0.02, f"exact-zero pixels {blk * 100:.1f}%")

# Per-image draws, or every view of one identity shares a framing and the
# augmentation teaches nothing about framing at all.
a_img, _ = scale_jitter(imgs, lm, 0.89, 1.14)
b_img, _ = scale_jitter(imgs, lm, 0.89, 1.14)
spread = float((a_img - b_img).abs().mean())
check("each image draws its own scale", spread > 1e-3,
      f"mean difference between two draws {spread:.4f}")


panel = torch.cat([torch.cat(list(imgs[:4].permute(0, 2, 3, 1)), 1),
                   torch.cat(list(vi[:4].permute(0, 2, 3, 1)), 1),
                   torch.cat(list(vi[N:N + 4].permute(0, 2, 3, 1)), 1)], 0)
Image.fromarray((panel.clamp(0, 1).numpy() * 255).astype(np.uint8)).save(OUT / "augment_test.png")
print("  wrote augment_test.png (original / view A / view B)")

print("")
print("=" * 56)
print("ALL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
