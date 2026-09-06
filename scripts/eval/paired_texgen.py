"""Paired per-face comparison of two texture generators.

An aggregate mean can move because a handful of faces changed a lot, which is
not the same claim as "this model is better". This scores BOTH models on the
SAME held-out faces and reports the paired difference: how many faces improved,
the median change, and a sign test. On ~900 faces that is a far better powered
comparison than NoW's 20 subjects.

    python scripts/eval/paired_texgen.py --a runs/texgen_gen6/model.pt \\
                                         --b runs/texgen_gen7/model.pt
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from face3d import assets
from face3d.geometry.flame_torch import FlameTorch
from face3d.geometry.landmarks import LandmarkEmbedding
from face3d.render.albedo import CACHE_DIR, FlameTexture
from face3d.texture.project import load_static
from face3d.texture.regions import crease_map, region_masks
from face3d.texture.skin import compose as skin_compose
from face3d.texture.skin import freckle_noise, split as skin_split
from face3d.texture.texgen import (TextureAutoencoder, TextureGenerator,
                                   diffuse_fill)

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(path, stage):
    ck = torch.load(path, map_location=DEV)
    if stage == "ae":
        m = TextureAutoencoder(ck["latent"], 256, ck["width"]).to(DEV)
    else:
        m = TextureGenerator(ck["latent"], 256, ck["width"], pretrained=False).to(DEV)
    m.load_state_dict(ck["model"], strict=False)
    m.eval()
    return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--cache", default=str(ROOT / "data" / "texgen_cache"))
    ap.add_argument("--stage", choices=("gen", "ae"), default="gen",
                    help="ae compares two autoencoders, i.e. two estimates of "
                         "the CEILING. They read the target, so they need the "
                         "same diffuse-filled construction training used.")
    ap.add_argument("--a-trained-below", type=int, default=0,
                    help="A trained on corpus indices < N, so val faces below "
                         "it flatter A. Shards sort old-first (shard_000.. "
                         "before shard_w0_..), so the old corpus occupies the "
                         "leading indices and this cleanly separates them.")
    args = ap.parse_args()

    cache = pathlib.Path(args.cache)
    d = {k: np.load(cache / f"{k}.npy", mmap_mode="r")
         for k in ("crop", "albedo", "weight", "coef", "eye")}
    n = len(d["crop"])
    perm = np.random.default_rng(0).permutation(n)
    val = perm[:max(64, int(0.05 * n))]

    ma, cka = load(args.a, args.stage)
    mb, ckb = load(args.b, args.stage)
    print(f"A {args.a}  epoch {cka['epoch']}  val {cka['val']:.5f}")
    print(f"B {args.b}  epoch {ckb['epoch']}  val {ckb['val']:.5f}")

    tex = FlameTexture(CACHE_DIR / "flame_texture_256_50.npz", device=DEV)
    flame = FlameTorch(assets.model_path("FLAME2023Open/flame2023_Open.pkl")).to(DEV)
    tex.attach_eyes(flame)
    eye_off = (1.0 - tex.eye_alpha).to(DEV)
    emb = LandmarkEmbedding(ROOT / "mediapipe_landmark_embedding" /
                            "mediapipe_landmark_embedding.npz", device=DEV)
    with np.load(CACHE_DIR / "flame_texture_256_50.npz") as dd:
        vt, ft = dd["vt"].astype(np.float32), dd["ft"].astype(np.int64)
    st = load_static(flame, vt, ft, resolution=256, device=DEV)
    masks = {k: v.to(DEV).float()
             for k, v in region_masks(flame, emb, st, device=DEV).items()}
    noise = torch.as_tensor(freckle_noise(256), device=DEV)
    crease = torch.as_tensor(crease_map(cache, st, 256), device=DEV)

    def score(model, crop, pca, alb, w, tgt):
        if args.stage == "ae":
            gen = (pca + model(tgt, w)).clamp(0, 1)
        else:
            res, raw = model(crop)
            gen = skin_compose((pca + res).clamp(0, 1), skin_split(raw),
                               masks, noise, crease)
        # Per FACE, not pooled: sum over texels and channels, divide by that
        # face's own weight, so a large face cannot dominate a small one.
        num = ((gen - alb).abs() * w).sum(dim=(1, 2, 3))
        den = (w.sum(dim=(1, 2, 3)) * 3).clamp(min=1e-6)
        return (num / den).cpu().numpy()

    ea, eb = [], []
    for lo in range(0, len(val), 16):
        j = np.sort(val[lo:lo + 16])
        alb = torch.from_numpy(d["albedo"][j].copy()).to(DEV).permute(0, 3, 1, 2).float() / 255
        w = torch.from_numpy(d["weight"][j].copy()).to(DEV).float().unsqueeze(1) / 255
        coef = torch.from_numpy(d["coef"][j].copy()).to(DEV)
        eye = torch.from_numpy(d["eye"][j].copy()).to(DEV)
        crop = torch.from_numpy(d["crop"][j].copy()).to(DEV).permute(0, 3, 1, 2).float() / 255
        w = w * eye_off
        with torch.no_grad():
            pca = tex.texture(coef, eye=eye).clamp(0, 1)
            rr = (alb - pca) * (w > 0)
            tgt = rr * w + diffuse_fill(rr, w) * (1 - w)
            ea.append(score(ma, crop, pca, alb, w, tgt))
            eb.append(score(mb, crop, pca, alb, w, tgt))

    ea, eb = np.concatenate(ea), np.concatenate(eb)
    order = np.sort(np.concatenate([np.sort(val[lo:lo + 16])
                                    for lo in range(0, len(val), 16)]))
    diff = eb - ea                      # negative = B better
    better = int((diff < 0).sum())
    # Sign test: under the null that each face is a coin flip, the count of
    # improvements is Binomial(n, 0.5). Normal approximation is ample at n~900.
    z = (better - len(diff) / 2) / (np.sqrt(len(diff)) / 2)

    print(f"\n{len(diff)} held-out faces, paired")
    print(f"  A mean {ea.mean():.5f}    B mean {eb.mean():.5f}    "
          f"{100*(1-eb.mean()/ea.mean()):+.2f}%")
    print(f"  median per-face change {np.median(diff):+.5f}")
    print(f"  B better on {better}/{len(diff)} faces ({100*better/len(diff):.1f}%)"
          f"   sign-test z = {z:.1f}")
    print(f"  B worse by >10% on {int((diff/ea > 0.10).sum())} faces; "
          f"better by >10% on {int((diff/ea < -0.10).sum())}")

    if args.a_trained_below:
        m = order >= args.a_trained_below
        if m.sum():
            da, db, dd = ea[m], eb[m], diff[m]
            bb = int((dd < 0).sum())
            zz = (bb - m.sum() / 2) / (np.sqrt(m.sum()) / 2)
            print(f"\n  restricted to the {int(m.sum())} faces NEITHER model "
                  f"trained on ({int((~m).sum())} of the split are A's own "
                  f"training data, which flatters A):")
            print(f"    A mean {da.mean():.5f}    B mean {db.mean():.5f}    "
                  f"{100*(1-db.mean()/da.mean()):+.2f}%")
            print(f"    B better on {bb}/{int(m.sum())} "
                  f"({100*bb/m.sum():.1f}%)   sign-test z = {zz:.1f}")


if __name__ == "__main__":
    main()
