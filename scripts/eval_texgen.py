"""Compare the three texture sources on held-out faces.

    python scripts/eval_texgen.py --model runs/texgen_gen/model.pt

Numbers and a contact sheet, because the two answer different questions. The
number says how close each source gets to the observed photograph; the sheet
says whether the failures are the kind a person minds. A model can win on masked
L1 by being smoothly wrong everywhere and still look worse than one that is
sharp and occasionally off.

The comparison is on the SAME held-out split the trainer used (seed 0), so the
generator has not seen any of these.
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d.albedo import CACHE_DIR, FlameTexture
from face3d.regions import crease_map, region_masks
from face3d.skin import compose as skin_compose
from face3d.skin import freckle_noise, split as skin_split
from face3d.texgen import (TextureAutoencoder, TextureGenerator,
                           diffuse_fill)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "runs" / "texgen_gen" / "model.pt"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "texgen_cache"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "out" / "texgen_eval.png"))
    ap.add_argument("--stage", choices=["ae", "gen"], default="gen",
                    help="ae reads the texture itself, so it measures whether "
                         "the latent can HOLD these textures; gen reads the "
                         "photograph, and is the thing that actually ships")
    a = ap.parse_args()

    cache = pathlib.Path(a.cache)
    d = {k: np.load(cache / f"{k}.npy", mmap_mode="r")
         for k in ("crop", "albedo", "weight", "coef", "eye")}
    n = len(d["crop"])
    perm = np.random.default_rng(0).permutation(n)
    val = perm[:max(64, int(0.05 * n))]

    ck = torch.load(a.model, map_location=DEV)
    if a.stage == "ae":
        model = TextureAutoencoder(ck["latent"], 256, ck["width"]).to(DEV)
    else:
        model = TextureGenerator(ck["latent"], 256, ck["width"],
                                 pretrained=False).to(DEV)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    # Checkpoints trained before the procedural layer have no params head. They
    # are still worth evaluating -- that is the whole point of comparing them --
    # so tolerate the gap and skip the composition rather than refuse to load.
    if missing:
        print(f"note: checkpoint predates {sorted(set(m.split('.')[0] for m in missing))}"
              f"; that part is skipped")
    if unexpected:
        raise RuntimeError(f"checkpoint has unexpected keys: {unexpected[:4]}")
    model.eval()
    if ck["stage"] != a.stage:
        print(f"WARNING: checkpoint is stage {ck['stage']}, asked for {a.stage}")
    print(f"{a.model}: stage {ck['stage']}  epoch {ck['epoch']}  val {ck['val']:.5f}")

    tex = FlameTexture(CACHE_DIR / "flame_texture_256_50.npz", device=DEV)
    from face3d import assets
    from face3d.flame_torch import FlameTorch
    flame_for_eyes = FlameTorch(
        assets.model_path("FLAME2023Open/flame2023_Open.pkl")).to(DEV)
    tex.attach_eyes(flame_for_eyes)
    eye_off = (1.0 - tex.eye_alpha).to(DEV)

    # The procedural layer, when the checkpoint has one. Older checkpoints
    # predate it and are still comparable -- they simply skip the composition.
    procedural = a.stage == "gen" and any(k.startswith("params.")
                                          for k in ck["model"])
    if procedural:
        from face3d.landmarks import LandmarkEmbedding
        from face3d.project import load_static
        emb = LandmarkEmbedding(ROOT / "mediapipe_landmark_embedding" /
                                "mediapipe_landmark_embedding.npz", device=DEV)
        with np.load(CACHE_DIR / "flame_texture_256_50.npz") as dd:
            vt, ft = dd["vt"].astype(np.float32), dd["ft"].astype(np.int64)
        st = load_static(flame_for_eyes, vt, ft, resolution=256, device=DEV)
        masks = {k: v.to(DEV).float() for k, v in
                 region_masks(flame_for_eyes, emb, st, device=DEV).items()}
        noise = torch.as_tensor(freckle_noise(256), device=DEV)
        crease = torch.as_tensor(
            crease_map(ROOT / "data" / "texgen_cache", st, 256), device=DEV)
    print(f"procedural skin layer: {procedural}")

    def predict(crop, tgt, wt, pca):
        if a.stage == "ae":
            return (pca + model(tgt, wt)).clamp(0, 1)
        res, raw = model(crop)
        base = (pca + res).clamp(0, 1)
        if not procedural:
            return base
        return skin_compose(base, skin_split(raw), masks, noise, crease)

    # --- numbers over the whole held-out split ---------------------------
    tot = {"pca": 0.0, "gen": 0.0}
    seen = 0
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
            # The autoencoder reads the TARGET, so it must be handed the same
            # construction training used -- measurement where observed, its own
            # smooth continuation elsewhere. Feeding it the old truncated
            # residual instead scored it at -3.5% against the basis, i.e. worse
            # than predicting nothing, purely from the input distribution
            # being wrong.
            rr = (alb - pca) * (w > 0)
            tgt = rr * w + diffuse_fill(rr, w) * (1 - w)
            gen = predict(crop, tgt, w, pca)
        # x3: the numerator sums three colour channels, so the weight
        # denominator must count each texel once per channel or every
        # figure here is inflated threefold.
        m = (w.sum() * 3).clamp(min=1e-6)
        tot["pca"] += float(((pca - alb).abs() * w).sum() / m) * len(j)
        tot["gen"] += float(((gen - alb).abs() * w).sum() / m) * len(j)
        seen += len(j)

    print(f"\nmasked L1 against the observed photograph, {seen} held-out faces")
    print(f"  PCA basis        {tot['pca']/seen:.5f}")
    print(f"  generator        {tot['gen']/seen:.5f}")
    gain = 100 * (1 - (tot["gen"] / seen) / max(tot["pca"] / seen, 1e-9))
    print(f"  generator is {gain:+.1f}% closer than the basis")

    # --- contact sheet ----------------------------------------------------
    S = 200
    rows = []
    for j in val[:a.n]:
        alb = torch.from_numpy(d["albedo"][j].copy()).to(DEV).permute(2, 0, 1)[None].float() / 255
        coef = torch.from_numpy(d["coef"][j].copy()).to(DEV)[None]
        eye = torch.from_numpy(d["eye"][j].copy()).to(DEV)[None]
        crop = torch.from_numpy(d["crop"][j].copy()).to(DEV).permute(2, 0, 1)[None].float() / 255
        w1 = (torch.from_numpy(d["weight"][j].copy()).to(DEV).float()[None, None]
              / 255) * eye_off
        with torch.no_grad():
            pca = tex.texture(coef, eye=eye).clamp(0, 1)
            rr1 = (alb - pca) * (w1 > 0)
            tgt1 = rr1 * w1 + diffuse_fill(rr1, w1) * (1 - w1)
            gen = predict(crop, tgt1, w1, pca)

        def im(t):
            return Image.fromarray(
                (t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).resize((S, S))
        rows.append([Image.fromarray(d["crop"][j].copy()).resize((S, S)),
                     im(pca), im(alb), im(gen)])

    sheet = Image.new("RGB", (4 * S, len(rows) * S), (18, 18, 18))
    for i, r in enumerate(rows):
        for k, t in enumerate(r):
            sheet.paste(t, (k * S, i * S))
    sheet.save(a.out)
    print(f"\nwrote {a.out}  [photo | PCA | projection (target) | GENERATOR]")


if __name__ == "__main__":
    main()
