"""Train the learned texture generator. Two stages -- see face3d/texgen.py.

    python scripts/train_texgen.py --stage ae                       # 1
    python scripts/train_texgen.py --stage gen --init runs/texgen_ae/model.pt

Stage 1 asks whether a latent of this size can hold these textures at all, by
giving the encoder the texture itself. Stage 2 asks whether that latent can be
predicted from the photograph. Running only stage 2 and getting a blurry result
tells you nothing about which of the two failed, which is the whole reason for
the split.

The target is a RESIDUAL over the PCA basis, rebuilt here from the stored
coefficients rather than stored per sample: 50 floats against 200 KB.
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d.albedo import CACHE_DIR, FlameTexture
from face3d.regions import crease_map, region_masks
from face3d.skin import compose, freckle_noise, region_means, split
from face3d.texgen import (TextureAutoencoder, TextureGenerator,
                           diffuse_fill, masked_loss)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_memmap(shard_dir, cache_dir):
    """Concatenate the npz shards into flat .npy files, once.

    npz is a zip: random access across shards means decompressing a whole shard
    to read one sample, which at shuffled batch order is most of a shard per
    item. Flat .npy can be memory-mapped, so the OS page cache does the work and
    the trainer never holds 2.5 GB of uint8 resident.

    Counted in a first pass over `coef` alone -- 50 floats a sample, so it
    decompresses 100 KB per shard instead of 200 MB -- which lets the output be
    allocated at exactly the right size. Allocating an overshoot and truncating
    afterwards needs a copy through RAM of the whole corpus, and on Windows the
    file cannot be replaced while a memmap still references it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    done = cache_dir / "index.npy"
    shards = sorted(pathlib.Path(shard_dir).glob("shard_*.npz"))
    if not shards:
        sys.exit(f"no shards in {shard_dir} -- run scripts/make_texture_corpus.py")

    if done.exists() and int(np.load(done)[1]) == len(shards):
        n = int(np.load(done)[0])
        print(f"memmap cache: {n} samples from {len(shards)} shards", flush=True)
        return n

    print(f"building memmap cache from {len(shards)} shards...", flush=True)
    counts = []
    for s_ in shards:
        with np.load(s_) as d:
            counts.append(len(d["coef"]))
    total = int(sum(counts))

    writers, off = {}, 0
    for s_, n in zip(shards, counts):
        with np.load(s_) as d:
            for key in ("crop", "albedo", "weight", "coef", "eye"):
                arr = d[key]
                if key not in writers:
                    writers[key] = np.lib.format.open_memmap(
                        cache_dir / f"{key}.npy", mode="w+", dtype=arr.dtype,
                        shape=(total,) + arr.shape[1:])
                writers[key][off:off + n] = arr
        off += n
    for w in writers.values():
        w.flush()
    writers.clear()
    np.save(done, np.array([total, len(shards)]))
    print(f"  {total} samples cached", flush=True)
    return total


class Corpus(torch.utils.data.Dataset):
    """Opens its memmaps lazily, per worker.

    Windows DataLoader workers are spawned, not forked, so the dataset is
    pickled into each one. An open np.memmap does not survive that, and holding
    the handles here would either fail to pickle or silently hand every worker
    the parent's file offsets. Opening on first access inside the worker gives
    each its own.
    """

    KEYS = ("crop", "albedo", "weight", "coef", "eye")

    def __init__(self, cache_dir, idx):
        self.cache_dir = pathlib.Path(cache_dir)
        self.idx = idx
        self.d = None

    def __len__(self):
        return len(self.idx)

    def _open(self):
        if self.d is None:
            self.d = {k: np.load(self.cache_dir / f"{k}.npy", mmap_mode="r")
                      for k in self.KEYS}
        return self.d

    def __getitem__(self, i):
        self.d = self._open()
        j = self.idx[i]
        return (torch.from_numpy(np.ascontiguousarray(self.d["crop"][j])),
                torch.from_numpy(np.ascontiguousarray(self.d["albedo"][j])),
                torch.from_numpy(np.ascontiguousarray(self.d["weight"][j])),
                torch.from_numpy(np.ascontiguousarray(self.d["coef"][j])),
                torch.from_numpy(np.ascontiguousarray(self.d["eye"][j])))


def prepare(batch, tex, eye_off, dev):
    """Batch -> (photo, residual target, loss weight, pca), all on device."""
    crop, alb, w, coef, eye = [t.to(dev, non_blocking=True) for t in batch]
    photo = crop.permute(0, 3, 1, 2).float() / 255.0
    alb = alb.permute(0, 3, 1, 2).float() / 255.0
    w = (w.float() / 255.0).unsqueeze(1)
    with torch.no_grad():
        pca = tex.texture(coef, eye=eye).clamp(0, 1)
    # Do not spend latent capacity on the eyeballs: the export composites a
    # procedurally generated iris over them regardless, so anything learned
    # there is overwritten. eye_off is 0 inside the eye discs.
    w = w * eye_off

    r = (alb - pca) * (w > 0)
    fill = diffuse_fill(r, w)
    # Continuous everywhere: the measurement where there is one, its own smooth
    # continuation where there is not. No step for the decoder to reproduce.
    target = r * w + fill * (1.0 - w)
    return photo, target, w, pca


def step(model, a, photo, target, w, pca, masks, noise, crease, colour_keys):
    """One forward pass and its loss, shared by train and val.

    Stage 1 has no procedural layer -- it exists to answer whether the latent
    can hold the textures, and adding a second mechanism to that question
    would make the answer un-attributable.

    Stage 2 is scored on the COMPOSED texture, not on the residual, because
    the composition is what ships. The colour parameters additionally get a
    direct target: the mean colour of the region they are named after. Without
    that they would be free to take any value that happened to reduce the image
    error, and "lip colour" would stop meaning lip colour -- which is the whole
    reason for having named parameters instead of another 11 latent dimensions.
    """
    if a.stage == "ae":
        pred = model(target, w)
        return masked_loss(pred.float(), target, w, off_weight=a.off_weight)

    res, raw = model(photo)
    p = split(raw.float())
    # NOT clamped before composing. The procedural layer darkens brows, lips
    # and creases, so the residual learns to pre-brighten them and let the
    # composition bring them back down. Clipping between the two breaks exactly
    # that cancellation: the residual is cut at 1.0, the darkening still
    # applies, and what survives is a white arc over each brow. compose()
    # clamps once at the end, which is the only place it is safe to.
    pred_tex = compose(pca + res.float(), p, masks, noise, crease)
    target_tex = (pca + target).clamp(0, 1)
    loss, l1, g = masked_loss(pred_tex, target_tex, w, off_weight=a.off_weight)

    # A light pull toward doing nothing. Cancellation between the two layers is
    # legitimate but it is also free, and left unpriced the residual drifts
    # into large opposing corrections that only agree where the masks are exact
    # -- so the seams of the procedural masks start to show in the residual.
    loss = loss + a.res_penalty * res.float().abs().mean()

    with torch.no_grad():
        want = region_means(target_tex, w, {k: masks[k] for k in
                                            ("skin", "lips", "brows")})
    aux = (  (p["skin"] - want["skin"]).abs().mean()
           + (p["lip"] - want["lips"]).abs().mean()
           + (p["brow"] - want["brows"]).abs().mean())
    return loss + a.aux_weight * aux, l1, g


def run(a):
    cache_dir = ROOT / "data" / "texgen_cache"
    n = build_memmap(ROOT / "data" / "texgen", cache_dir)

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = max(64, int(0.05 * n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"train {len(train_idx)}  val {len(val_idx)}", flush=True)

    tex = FlameTexture(CACHE_DIR / "flame_texture_256_50.npz", device=DEV)
    from face3d import assets
    from face3d.flame_torch import FlameTorch
    flame = FlameTorch(assets.model_path("FLAME2023Open/flame2023_Open.pkl")).to(DEV)
    tex.attach_eyes(flame)
    eye_off = (1.0 - tex.eye_alpha).to(DEV)                 # (1,1,256,256)

    # The procedural layer's fixed inputs. Topology and corpus only, so they
    # are built once and shared by every sample.
    from face3d.landmarks import LandmarkEmbedding
    from face3d.project import load_static
    emb = LandmarkEmbedding(ROOT / "mediapipe_landmark_embedding" /
                            "mediapipe_landmark_embedding.npz", device=DEV)
    with np.load(CACHE_DIR / "flame_texture_256_50.npz") as d:
        vt, ft = d["vt"].astype(np.float32), d["ft"].astype(np.int64)
    static = load_static(flame, vt, ft, resolution=256, device=DEV)
    masks = {k: v.to(DEV).float() for k, v in
             region_masks(flame, emb, static, device=DEV).items()}
    noise = torch.as_tensor(freckle_noise(256), device=DEV)
    crease = torch.as_tensor(crease_map(cache_dir, static, 256), device=DEV)
    colour_keys = ("skin", "lip", "brow")

    mk = lambda idx, sh: torch.utils.data.DataLoader(
        Corpus(cache_dir, idx), batch_size=a.batch, shuffle=sh,
        num_workers=a.workers, pin_memory=(DEV == "cuda"), drop_last=sh)
    train_dl, val_dl = mk(train_idx, True), mk(val_idx, False)

    if a.stage == "ae":
        model = TextureAutoencoder(a.latent, 256, a.width).to(DEV)
    else:
        model = TextureGenerator(a.latent, 256, a.width, pretrained=True).to(DEV)
        if a.init:
            sd = torch.load(a.init, map_location=DEV)["model"]
            dec = {k[4:]: v for k, v in sd.items() if k.startswith("dec.")}
            model.dec.load_state_dict(dec)
            print(f"decoder warm-started from {a.init}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(train_dl))
    out = ROOT / "runs" / (a.out or f"texgen_{a.stage}")
    out.mkdir(parents=True, exist_ok=True)
    amp = DEV == "cuda"

    best = float("inf")
    for ep in range(a.epochs):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for batch in train_dl:
            photo, target, w, pca = prepare(batch, tex, eye_off, DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss, l1, g = step(model, a, photo, target, w, pca,
                                   masks, noise, crease, colour_keys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(photo)
            seen += len(photo)

        model.eval()
        vtot, vseen, vl1 = 0.0, 0, 0.0
        with torch.no_grad():
            for batch in val_dl:
                photo, target, w, pca = prepare(batch, tex, eye_off, DEV)
                loss, l1, g = step(model, a, photo, target, w, pca,
                                   masks, noise, crease, colour_keys)
                vtot += float(loss) * len(photo)
                vl1 += float(l1) * len(photo)
                vseen += len(photo)

        v = vtot / max(vseen, 1)
        print(f"epoch {ep+1:3d}/{a.epochs}  train {tot/max(seen,1):.5f}  "
              f"val {v:.5f}  (L1 {vl1/max(vseen,1):.5f})  "
              f"{time.time()-t0:.0f}s", flush=True)
        if v < best:
            best = v
            torch.save({"model": model.state_dict(), "stage": a.stage,
                        "latent": a.latent, "width": a.width, "epoch": ep,
                        "val": v}, out / "model.pt")
    print(f"best val {best:.5f} -> {out/'model.pt'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ae", "gen"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--off-weight", type=float, default=0.35)
    ap.add_argument("--res-penalty", type=float, default=0.08,
                    help="price on the learned residual, so it does not "
                         "pre-compensate for the procedural layer")
    ap.add_argument("--aux-weight", type=float, default=0.5,
                    help="pull on the named colour parameters toward the mean "
                         "colour of the region they are named after")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--init", default="")
    ap.add_argument("--out", default="")
    run(ap.parse_args())
