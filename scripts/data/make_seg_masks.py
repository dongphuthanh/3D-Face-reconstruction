"""Precompute face-skin masks for a corpus, so training can gate the
photometric loss on them.

Why cache rather than segment in the loop: MediaPipe runs on CPU at roughly
25 ms an image, so a batch of 32 would cost ~0.8 s against a ~30 ms training
step. Segmenting once and storing the result turns a 30x slowdown into a
one-off pass.

Stored at 112x112 and bit-packed. The mask weights a loss rather than
selecting pixels exactly, so half resolution is not a meaningful loss of
precision, and it keeps FFHQ at ~30 MB instead of ~500 MB.

    python scripts/data/make_seg_masks.py --corpus ffhq
    python scripts/data/make_seg_masks.py --corpus digiface --limit 40000
"""

import argparse
import pathlib
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from face3d.texture.segment import FACE_SKIN, Segmenter

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = 112


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="ffhq")
    ap.add_argument("--limit", type=int, default=0, help="0 = every crop")
    ap.add_argument("--res", type=int, default=RES)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    crops = ROOT / "data" / a.corpus / "crops"
    if not crops.exists():
        print(f"SKIP - no crops under {crops}")
        return
    files = sorted(crops.rglob("*.jpg")) + sorted(crops.rglob("*.png"))
    if a.limit:
        files = files[:a.limit]
    out = pathlib.Path(a.out or (ROOT / "data" / a.corpus / f"skin_{a.res}.npz"))

    seg = Segmenter()
    keys, packed = [], []
    t0 = time.time()
    for i, f in enumerate(files):
        img = np.asarray(Image.open(f).convert("RGB").resize((256, 256)))
        m = (seg.categories(img) == FACE_SKIN)
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255)
                       .resize((a.res, a.res), Image.BILINEAR)) > 127
        keys.append(str(f.relative_to(crops)).replace("\\", "/"))
        packed.append(np.packbits(m.ravel()))
        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(files)}  {rate:.0f} img/s  "
                  f"eta {(len(files)-i-1)/rate/60:.1f} min", flush=True)

    np.savez_compressed(out, keys=np.array(keys), masks=np.stack(packed),
                        res=a.res)
    mb = out.stat().st_size / 1e6
    print(f"\nwrote {out}  ({len(keys)} masks, {mb:.1f} MB, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
