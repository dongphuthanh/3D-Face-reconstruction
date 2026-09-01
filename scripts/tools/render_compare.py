"""Full-fidelity render against the original photograph.

Everything the model predicts is used -- identity shape (calibrated, as
shipped), expression, jaw and neck pose, camera, spherical-harmonic light and
albedo. This is what the product actually outputs, unlike the neutral-mesh
views NoW and diag_corpus_identity.py work with.

Read the ISOLATED RENDER column, not the overlay. The overlay keeps the real
hair, ears, jawline and background, and the eye takes identity from those --
a mean face composited into the same photograph already looks like the person.
Measured on 61 held-out NoW subjects, facenet cosine to the photo:

    model         variant             overlay   render alone
    deca_celswap  predicted shape      0.669       0.479
    deca_celswap  MEAN FACE control    0.548       0.375
    deca_joint    predicted shape      0.669       0.489
    deca_joint    MEAN FACE control    0.557       0.381

The control is the same pipeline with identity shape zeroed and expression,
pose, light and albedo kept, so the gap is geometry alone: a consistent +0.11.
Same-person pairs usually score 0.7-0.9, so 0.49 is recognisable-ish, not
confident.

Input quality dominates. A clean frontal studio portrait scored 0.715 with a
+0.226 geometry gain -- roughly twice the population figure -- while NoW's set
includes profiles, expressions and occlusions and only 61 of 100 subjects had a
detectable face in their first image. Controlling capture conditions is worth
more here than any training change measured in face3d/learn/encoder.py.
"""
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from face3d import assets
from face3d.render.albedo import CACHE_DIR, FlameTexture
from face3d.learn.encoder import ResNetEncoder
from face3d.geometry.facemask import face_faces, face_region
from face3d.geometry.flame_torch import FlameTorch
from face3d.geometry.landmarks import LandmarkEmbedding
from face3d.render.pipeline import FaceRenderer

DEV = "cuda"
ARMS = ["deca_celswap", "deca_joint"]

flame = FlameTorch(assets.model_path_or_skip()).to(DEV)
emb = LandmarkEmbedding(ROOT / "mediapipe_landmark_embedding" /
                        "mediapipe_landmark_embedding.npz", device=DEV)
tc = CACHE_DIR / "flame_texture_256_50.npz"
tex = FlameTexture(tc, device=DEV) if tc.exists() else None
assert tex is not None, "texture cache missing; render would be flat grey"
keep = face_faces(flame, face_region(flame, emb, radius=0.045))
r = FaceRenderer(flame, lmk_idx=emb, image_size=224, texture=tex, face_keep=keep)

# Spread across FFHQ for variety in age, sex and skin tone.
z = np.load(ROOT / "data" / "ffhq" / "landmarks_224.npz")
avail = []
for k in [str(x) for x in z["keys"]]:
    p = ROOT / "data" / "ffhq" / "crops" / f"{k}.jpg"
    if p.exists():
        avail.append(p)
    if len(avail) >= 1400:
        break
sel = [avail[i] for i in (2, 3, 180, 420, 900, 1150, 1300)]
arrs = [np.asarray(Image.open(p).convert("RGB")) for p in sel]
x = torch.from_numpy(np.stack(arrs)).to(DEV).permute(0, 3, 1, 2).float() / 255.0


def u8(t):
    return (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


panels = {}
for arm in ARMS:
    ck = ROOT / "runs" / arm / "encoder.pt"
    blob = torch.load(ck, map_location=DEV)
    enc = ResNetEncoder().to(DEV)
    enc.load_state_dict(blob["model"])
    enc.eval()
    with torch.no_grad():
        # calibrate=True is the shipped path: predict() returns the raw
        # over-confident shape and export_head scales it by SHAPE_CALIBRATION.
        p = enc.predict(x, calibrate=True)
        v, _ = r.geometry(p)
        img, mask = r.render(v, p)
    ren = [u8(img[i]) for i in range(len(arrs))]
    m = [mask[i].cpu().numpy()[..., None] for i in range(len(arrs))]
    over = [(arrs[i] * (1 - m[i]) + ren[i] * m[i]).astype(np.uint8)
            for i in range(len(arrs))]
    panels[arm] = (ren, over)
    print(f"  rendered {arm}")

rows = []
for i in range(len(arrs)):
    row = [arrs[i]]
    for arm in ARMS:
        row += [panels[arm][0][i], panels[arm][1][i]]
    rows.append(np.concatenate(row, 1))
out = ROOT / "out" / "likeness_grid.png"
Image.fromarray(np.concatenate(rows, 0)).save(out)
print(f"\nphoto | celswap render | celswap overlay | joint render | joint overlay")
print(f"-> {out}")
