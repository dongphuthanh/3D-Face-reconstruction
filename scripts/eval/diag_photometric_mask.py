"""How much of the photometric loss lands on pixels that are not face skin?

The photometric mask is the rendered mesh silhouette restricted to face-skin
TRIANGLES. That is mesh-side: it selects which triangles to compare, not which
photograph pixels are actually skin. An occluder inside the projected face --
spectacle frames, a fringe, a hand -- still enters the loss, and since the
albedo model can only paint skin, GEOMETRY is the free variable left to explain
those pixels.

This measures the size of that leak before anyone spends a training run on it:
segment the crop, and report what fraction of the pixels the loss currently
counts are not FACE_SKIN.

    python scripts/eval/diag_photometric_mask.py --n 200
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from face3d import assets
from face3d.geometry.facemask import face_faces, face_region
from face3d.geometry.flame_torch import FlameTorch
from face3d.geometry.landmarks import LandmarkEmbedding
from face3d.learn.encoder import ResNetEncoder
from face3d.render.albedo import CACHE_DIR, FlameTexture
from face3d.render.pipeline import FaceRenderer
from face3d.texture.segment import BODY_SKIN, FACE_SKIN, HAIR, OTHER, CLOTHES, Segmenter

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NAMES = {0: "background", 1: "hair", 2: "body skin", 3: "face skin",
         4: "clothes", 5: "other"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="ffhq", choices=("ffhq", "digiface"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--checkpoint", default=str(ROOT / "runs" / "deca_open" / "encoder.pt"))
    ap.add_argument("--flame-model", default="FLAME2023Open/flame2023_Open.pkl")
    ap.add_argument("--skin-mask", type=float, default=0.045)
    a = ap.parse_args()

    crops = ROOT / "data" / a.corpus / "crops"
    files = sorted(crops.rglob("*.jpg"))[:a.n] or sorted(crops.rglob("*.png"))[:a.n]
    if not files:
        print(f"SKIP - no crops under {crops}")
        return
    ckpt = pathlib.Path(a.checkpoint)
    if not ckpt.exists():
        print(f"SKIP - no checkpoint at {ckpt}")
        return

    flame = FlameTorch(assets.model_path(a.flame_model)).to(DEV)
    emb = LandmarkEmbedding(
        ROOT / "mediapipe_landmark_embedding" / "mediapipe_landmark_embedding.npz",
        device=DEV)
    tex_cache = CACHE_DIR / "flame_texture_256_50.npz"
    tex = FlameTexture(tex_cache, device=DEV) if tex_cache.exists() else None
    if tex is not None:
        tex.attach_eyes(flame)
    blob = torch.load(ckpt, map_location=DEV)
    enc = ResNetEncoder(n_shape=100, n_expr=50, pretrained=False).to(DEV)
    enc.load_state_dict(blob["model"])
    enc.eval()

    # Exactly the mask the training loop builds: face-skin triangles only.
    keep = face_faces(flame, face_region(flame, emb, radius=a.skin_mask))
    renderer = FaceRenderer(flame, lmk_idx=emb, image_size=224, texture=tex,
                            face_keep=keep)
    seg = Segmenter()

    from PIL import Image
    tally = np.zeros(6, dtype=np.int64)
    per_image = []
    for f in files:
        img = np.asarray(Image.open(f).convert("RGB").resize((224, 224)))
        x = torch.from_numpy(img).to(DEV).float().permute(2, 0, 1)[None] / 255.0
        with torch.no_grad():
            p = enc.predict(x, calibrate=True)
            verts, _ = renderer.geometry(p)
            _, mask = renderer.render(verts, p)
        m = mask[0].cpu().numpy().astype(bool)          # what the loss counts
        if m.sum() == 0:
            continue
        cats = seg.categories(img)
        counts = np.bincount(cats[m], minlength=6)
        tally += counts
        per_image.append(1.0 - counts[FACE_SKIN] / counts.sum())

    total = tally.sum()
    print(f"\n{a.corpus}: {len(per_image)} images, "
          f"{total/1e6:.1f} M pixels inside the photometric mask\n")
    print(f"{'class':<12}{'share of loss pixels':>22}")
    print("-" * 34)
    for i in np.argsort(-tally):
        print(f"{NAMES[i]:<12}{100*tally[i]/total:>21.2f}%")
    leak = 100 * (1 - tally[FACE_SKIN] / total)
    per_image = np.array(per_image)
    print("-" * 34)
    print(f"{'NOT face skin':<12}{leak:>21.2f}%   <- the leak")
    print(f"\nper-image leak: median {100*np.median(per_image):.1f}%  "
          f"p90 {100*np.quantile(per_image, 0.9):.1f}%  "
          f"max {100*per_image.max():.1f}%")
    print(f"images with >20% leak: "
          f"{int((per_image > 0.20).sum())}/{len(per_image)}")


if __name__ == "__main__":
    main()
