"""Precompute ArcFace identity embeddings for ingested crops.

MICA's insight: a face-recognition network has already solved "who is this",
having been trained on millions of faces to encode identity while discarding
pose, lighting and expression -- exactly the nuisance factors our encoder keeps
absorbing into shape. So rather than learning identity from a 1-2 px signal in
pixels, feed the embedding in and learn only the small mapping to FLAME shape.

Measured on NoW validation, between/within subject spread:

    our encoder's shape output   0.76
    ArcFace embedding            1.56

Twice the identity separation, and above 1.0 -- identity dominates, which our
encoder has never managed on real photographs.

Detection is skipped: the crops are already face-centred, and running the
detector again costs 270 ms/image against 42 ms for recognition alone. The
trade is that ArcFace sees our MediaPipe framing rather than its own 5-point
alignment, which loses some accuracy -- acceptable, since both training and
inference use the same framing.

Embeddings are stored float16. They are unit-normalised, so the precision is
ample and it halves a 1 GB array.
"""

import argparse
import os
import pathlib
import sys
import time
import warnings

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "digiface"))
    ap.add_argument("--identities", type=int, default=10000,
                    help="how many identities to embed; 0 for all")
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()

    root = pathlib.Path(a.data)
    cache = root / f"landmarks_{a.size}.npz"
    if not cache.exists():
        print(f"SKIP - no ingest at {cache}")
        sys.exit(0)

    z = np.load(cache)
    keys = np.array([str(k) for k in z["keys"]])
    subj = np.array([str(s) for s in z["subject"]])

    if a.identities:
        wanted = set(sorted(set(subj.tolist()))[: a.identities])
        sel = np.array([s in wanted for s in subj])
        keys, subj = keys[sel], subj[sel]

    out = root / f"arcface_{a.size}.npz"
    have = {}
    if out.exists():
        zc = np.load(out)
        have = {str(k): v for k, v in zip(zc["keys"], zc["embeddings"])}
        print(f"    reusing {len(have)} cached embeddings")

    todo = [k for k in keys if k not in have]
    print(f"=== ArcFace embeddings: {len(keys)} crops, {len(todo)} to compute ===")
    if not todo:
        print("  nothing to do")
        return

    from insightface.model_zoo import get_model
    model = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
    if not os.path.exists(model):
        print(f"SKIP - ArcFace weights missing at {model}. They download on first "
              f"use of insightface.app.FaceAnalysis(name='buffalo_l').")
        sys.exit(0)
    rec = get_model(model)
    rec.prepare(ctx_id=0)

    crops = root / "crops"
    t0 = time.time()
    for i, k in enumerate(todo, 1):
        im = Image.open(crops / f"{k}.jpg").convert("RGB").resize((112, 112))
        # get_feat expects BGR, as insightface does throughout
        have[k] = rec.get_feat(np.asarray(im)[:, :, ::-1]).ravel().astype(np.float16)
        if i % 2000 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el:.0f}s  {i / max(el, 1):.1f}/s  "
                  f"eta {(len(todo) - i) / max(i / max(el, 1), 1e-9) / 60:.0f} min",
                  flush=True)

    ks = [k for k in have]
    np.savez_compressed(out, keys=np.array(ks),
                        embeddings=np.stack([have[k] for k in ks]))
    print(f"\n  wrote {out.name}  ({len(ks)} embeddings, "
          f"{out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
