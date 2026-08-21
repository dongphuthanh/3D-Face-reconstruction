"""Does the encoder encode identity at all?

One number answers it: the ratio of between-subject to within-subject spread in
predicted shape coefficients. If the same person photographed twice yields more
variation than two different people do, the encoder is responding to pose and
lighting rather than to who it is looking at.

    ratio < 1   no identity signal
    ratio ~ 1   identity and nuisance factors equally strong
    ratio > 1   identity dominates

The first FFHQ run measured 0.41. That single number distinguishes "learned
nothing about identity" from "learned identity badly", which the NoW score alone
cannot -- both look like a mediocre millimetre figure.
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d import assets, now
from face3d.detect import FaceDetector, crop_square
from face3d.encoder import ResNetEncoder
from face3d.flame_torch import FlameTorch

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES = ROOT / "NoW_Dataset" / "final_release_version" / "iphone_pictures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs" / "ffhq" / "encoder.pt"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--crop-margin", type=float, default=1.6,
                    help="must match the margin the model was ingested with. "
                         "Arc2Face used 1.15 (its sources are pre-cropped); FFHQ "
                         "and DigiFace used 1.6. A mismatch is a distribution "
                         "shift that shows up as inflated deviation and a "
                         "collapsed identity ratio")
    ap.add_argument("--identity-data", default="",
                    help="measure on held-out identities from an identity-grouped "
                         "ingest instead of NoW. Separates 'did it learn identity' "
                         "(in-domain) from 'does it transfer to photographs'")
    a = ap.parse_args()

    ckpt = pathlib.Path(a.checkpoint)
    if not ckpt.exists():
        print(f"SKIP - no checkpoint at {ckpt}")
        sys.exit(0)
    if not a.identity_data and not IMAGES.exists():
        print("SKIP - needs the NoW images, or --identity-data")
        sys.exit(0)

    flame = FlameTorch(assets.model_path_or_skip()).to(DEV)
    enc = ResNetEncoder(n_shape=100, n_expr=50, pretrained=False).to(DEV)
    enc.load_state_dict(torch.load(ckpt, map_location=DEV)["model"])
    enc.eval()

    shapes, subjects = [], []
    if a.identity_data:
        # Crops are already detected and cropped at ingest, so no detector here.
        from face3d.data import IdentityPairs
        ds = IdentityPairs(pathlib.Path(a.identity_data), 224, "val")
        n = 0
        for k in range(len(ds)):
            g = ds.groups[k]
            for i in g:
                im = Image.open(ds.dir / f"{ds.keys[i]}.jpg").convert("RGB")
                x = torch.from_numpy(np.array(im, np.uint8)).permute(2, 0, 1)[None]
                with torch.no_grad():
                    p = enc.predict(x.float().to(DEV) / 255.0)
                shapes.append(p.shape[0].cpu().numpy())
                subjects.append(str(k))
                n += 1
            if n >= a.limit:
                break
    else:
        det = FaceDetector(blendshapes=False)
        for rel in now.image_list(ROOT, "validation")[: a.limit]:
            im = np.asarray(Image.open(IMAGES / rel).convert("RGB"))
            r = det.detect(im)
            if r is None:
                continue
            crop, _ = crop_square(im, r["norm"], size=224, margin=a.crop_margin)
            x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)[None]
            with torch.no_grad():
                p = enc.predict(x.float().to(DEV) / 255.0)
            shapes.append(p.shape[0].cpu().numpy())
            subjects.append(rel.split("/")[0])
        det.close()

    S = np.stack(shapes)
    subj = np.array(subjects)
    withins, means = [], []
    for s in sorted(set(subjects)):
        g = S[subj == s]
        if len(g) < 2:
            continue
        means.append(g.mean(0))
        withins.append(np.linalg.norm(g - g.mean(0), axis=1).mean())
    means = np.stack(means)
    within = float(np.mean(withins))
    between = float(np.linalg.norm(means - means.mean(0), axis=1).mean())

    with torch.no_grad():
        v0, _ = flame(batch_size=1)
        full = torch.zeros(len(S), flame.n_shape, device=DEV)
        full[:, :100] = torch.tensor(S, device=DEV)
        v, _ = flame(full)
        dev_mm = float((v - v0).norm(dim=-1).mean().item() * 1000)

    print(f"checkpoint: {ckpt}"
          + (f"   [in-domain: {a.identity_data}]" if a.identity_data
             else "   [NoW photographs]"))
    print(f"  {len(S)} images across {len(means)} subjects")
    print("")
    print(f"  within-subject  spread  {within:.3f}")
    print(f"  between-subject spread  {between:.3f}")
    print(f"  ratio                   {between / max(within, 1e-9):.2f}"
          "   (>1 means identity dominates)")
    print(f"  mesh deviation from FLAME mean  {dev_mm:.2f} mm")


if __name__ == "__main__":
    main()
