"""Build the training corpus for the learned texture generator.

Runs the gated projection over FFHQ and stores, per subject, what a texture
model needs to learn from:

    crop    (224,224,3) uint8   the photograph the encoder saw
    albedo  (256,256,3) uint8   the projected texture, occluder-gated
    weight  (256,256)   uint8   per-texel confidence; 0 = never observed
    coef    (50,)       float32 PCA albedo coefficients for the same subject
    eye     (6,)        float32 iris and sclera colours

`weight` is the important one and the reason this is not just a folder of
images. Every projected texture has holes -- the back of the head, anything
occluded, anything the segmenter rejected -- and a model trained without
masking those would learn to reproduce the fallback, which is the basis mean
and the mirror fill. It would be learning our own guesswork back.

`coef` is stored rather than the reconstructed basis texture because 50 floats
are 200 bytes and the texture is 200 KB. The trainer rebuilds it.

FFHQ only, deliberately. DigiFace is 26x larger but synthetic: its skin is
rendered, and a texture model trained on it would learn rendered pores. This is
the one place in the project where the real corpus is unambiguously the right
one, and it is also the licence-permissive subset already in deca_open's
lineage.

    python scripts/data/make_texture_corpus.py --limit 6000
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from face3d.learn.detect import crop_square
from face3d.texture.segment import FACE_SKIN
from face3d.texture.project import project_photo
from webapp.pipeline import (CROP_MARGIN, PROJECT_CROP, PROJECT_SCREEN,
                             SEG_SIZE, Reconstructor)

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = 256          # stored texture resolution; the generator works at this size

# Below this fraction of the fitted face region observed, the sample teaches
# more about our mirror fill than about the person. A profile shot with one
# cheek entirely unseen is the usual cause.
MIN_COVERAGE = 0.45


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "data" / "ffhq" / "crops"))
    ap.add_argument("--out", default=str(ROOT / "data" / "texgen"))
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--shard", type=int, default=500)
    a = ap.parse_args()

    from PIL import Image

    rec = Reconstructor()
    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(pathlib.Path(a.images).glob("*.jpg"))[:a.limit]
    print(f"{len(paths)} candidates from {a.images}", flush=True)

    face_area = float((rec.face_mask > 0.5).sum())
    buf, shard, kept, skipped = [], 0, 0, 0
    t0 = time.time()

    for i, path in enumerate(paths):
        try:
            img = np.asarray(Image.open(path).convert("RGB"))
            res = rec._detector().detect(img)
            if res is None:
                skipped += 1
                continue

            crop, _ = crop_square(img, res["norm"], size=224, margin=CROP_MARGIN)
            x = torch.from_numpy(np.ascontiguousarray(crop))
            x = x.permute(2, 0, 1)[None].float() / 255.0

            with torch.no_grad():
                p = rec.enc.predict(x, calibrate=True)
                pp = p.pad_to(rec.flame.n_shape, rec.flame.n_expr)
                verts, _ = rec.flame(pp.shape, pp.expr, pp.pose)

                hi, _ = crop_square(img, res["norm"], size=PROJECT_CROP,
                                    margin=CROP_MARGIN)
                photo = torch.from_numpy(np.ascontiguousarray(hi)).float() / 255.0

                seg, _ = crop_square(img, res["norm"], size=SEG_SIZE,
                                     margin=CROP_MARGIN)
                allow = rec._segmenter().categories(seg) == FACE_SKIN
                pm = rec._soften(allow.astype(np.float32), erode=0.0, blur=0.004)

                alb, w = project_photo(rec.flame, photo, p, verts, rec.static,
                                       face_mask=rec.proj_mask,
                                       screen=PROJECT_SCREEN, photo_mask=pm,
                                       facing_min=0.15, facing_max=0.50)

            # Coverage is measured against the FITTED face region, not the whole
            # map: the scalp and neck are never observed and would drag every
            # sample below any sensible threshold.
            cov = float((w.cpu().numpy() * (rec.face_mask > 0.5)).sum()) / face_area
            if cov < MIN_COVERAGE:
                skipped += 1
                continue

            def down(t, ch):
                t = t.permute(2, 0, 1)[None] if ch == 3 else t[None, None]
                t = F.interpolate(t, size=(RES, RES), mode="area")
                return (t[0].permute(1, 2, 0) if ch == 3 else t[0, 0])

            # Zero the unobserved region rather than storing what the raw
            # projection leaves there, which is background, duplicated faces and
            # whatever the mirror fill invented. It is masked out by `weight`
            # either way, but a corpus you can open and read is worth having,
            # and zeros compress.
            alb_s = down(alb, 3).clamp(0, 1)
            w_s = down(w, 1).clamp(0, 1)
            alb_s = alb_s * (w_s > 0.02).to(alb_s.dtype)[..., None]

            buf.append(dict(
                crop=crop.astype(np.uint8),
                albedo=(alb_s.cpu().numpy() * 255).astype(np.uint8),
                weight=(w_s.cpu().numpy() * 255).astype(np.uint8),
                coef=p.albedo[0].cpu().numpy().astype(np.float32),
                eye=p.eye[0].cpu().numpy().astype(np.float32),
            ))
            kept += 1
        except Exception as e:                    # one bad file must not stop 6000
            print(f"  skip {path.name}: {type(e).__name__}: {e}", flush=True)
            skipped += 1
            continue

        if len(buf) >= a.shard:
            write(out_dir, shard, buf)
            shard += 1
            buf = []
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            eta = (len(paths) - i - 1) / max(rate, 1e-6) / 60
            print(f"  {kept} kept / {skipped} skipped / {i+1} seen  "
                  f"{rate:.1f} img/s  eta {eta:.0f} min", flush=True)

    if buf:
        write(out_dir, shard, buf)
    print(f"done: {kept} kept, {skipped} skipped, {shard + bool(buf)} shards "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)


def write(out_dir, idx, buf):
    path = out_dir / f"shard_{idx:03d}.npz"
    np.savez(path,
             crop=np.stack([b["crop"] for b in buf]),
             albedo=np.stack([b["albedo"] for b in buf]),
             weight=np.stack([b["weight"] for b in buf]),
             coef=np.stack([b["coef"] for b in buf]),
             eye=np.stack([b["eye"] for b in buf]))
    print(f"  wrote {path.name}  ({len(buf)} samples, "
          f"{path.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
