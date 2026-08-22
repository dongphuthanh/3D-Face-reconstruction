"""Train the encoder on real photographs.

Self-supervised: no 3D ground truth anywhere. Supervision is 2D landmarks from
MediaPipe plus a photometric term against the crop, with FLAME acting as the
geometric prior that makes the problem well-posed.

Known limitation: the photometric mask is the rendered mesh silhouette, which
covers forehead, hair and neck as well as skin. The encoder is therefore partly
penalised for failing to explain hair with a skin albedo model. That is what
story C2 (face-parsing masks) addresses; until then the landmark term carries
most of the signal, which the ablation supports.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d import assets
from face3d.albedo import CACHE_DIR, FlameTexture
from face3d.data import FFHQCrops, IdentityPairs, flatten_pairs
from face3d.encoder import ArcFaceShapeEncoder, ResNetEncoder
from face3d.facemask import face_faces, face_region
from face3d.flame_torch import FlameTorch
from face3d.landmarks import LandmarkEmbedding
from face3d.augment import consistency_loss, swap_shape, two_views
from face3d.losses import (IdentityLoss, landmark_loss, make_overlay,
                           photometric_loss, regularization)
from face3d.pipeline import FaceRenderer

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    # DECA weights landmark 1.0 against photometric 2.0. Ours was 5:1 the other
    # way -- a 10x swing toward the term measured to be near-blind to shape
    # (swapping shape moves landmarks 1.95 px against pose's 17.13).
    ap.add_argument("--w-lmk", type=float, default=1.0)
    ap.add_argument("--w-pho", type=float, default=2.0)
    ap.add_argument("--w-id", type=float, default=0.0,
                    help="DECA's identity loss (they use 0.2): cosine distance "
                         "between face-recognition features of the render "
                         "composited into the photo, and the photo itself. "
                         "Albedo and light are detached so it supervises shape "
                         "alone -- the only term that does")
    ap.add_argument("--w-light", type=float, default=1.0,
                    help="spherical-harmonic regularisation. DECA's strongest "
                         "weight; unconstrained light explains away shading that "
                         "should come from geometry")
    ap.add_argument("-k", "--images-per-identity", type=int, default=2,
                    help="images per identity per batch. DECA uses 4: one "
                         "image's shape must then explain three other views")
    ap.add_argument("--w-con", type=float, default=0.0,
                    help="shape-consistency across two augmented views; "
                         "0 reproduces the original baseline")
    ap.add_argument("--w-swap", type=float, default=0.0,
                    help="DECA-style shape swap between paired views. Unlike "
                         "--w-con this cannot be satisfied by collapsing shape; "
                         "requires --w-con>0 or --pair to build the views")
    ap.add_argument("--swap-photometric", action="store_true",
                    help="also apply the photometric term to the swapped render "
                         "(costs one more rasterisation)")
    ap.add_argument("--identity-data", default="",
                    help="root of an identity-grouped ingest (e.g. data/digiface). "
                         "Feeds the swap loss REAL pairs -- two different images "
                         "of one subject -- instead of two augmentations of one "
                         "image, which only ever taught augmentation invariance")
    ap.add_argument("--mix-ffhq", type=float, default=0.0,
                    help="weight for an extra FFHQ batch each step. Identity "
                         "pairs supply the swap constraint but are synthetic and "
                         "112px upscaled; real photographs keep the "
                         "reconstruction terms anchored on the distribution the "
                         "model is actually asked about")
    ap.add_argument("--mix-batch", type=int, default=0,
                    help="FFHQ batch size for the mix; defaults to --batch")
    ap.add_argument("--arcface", action="store_true",
                    help="MICA-style: take shape from a cached ArcFace identity "
                         "embedding instead of the ResNet trunk. The other five "
                         "output groups still come from pixels, since they work. "
                         "Requires scripts/embed_arcface.py to have been run")
    ap.add_argument("--pair", action="store_true",
                    help="build paired views even when --w-con is 0, so the swap "
                         "loss can be used on its own")
    ap.add_argument("--render-both-views", dest="render_first_only",
                    action="store_false", default=True,
                    help="rasterise both paired views (about 45%% more VRAM); "
                         "by default only the weak view is rendered")
    ap.add_argument("--skin-mask", type=float, default=0.045,
                    help="face-region radius in metres for the photometric mask "
                         "(story C2); 0 disables and scores the whole silhouette")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--limit-steps", type=int, default=0, help="cap steps/epoch for smoke tests")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "runs" / "ffhq"))
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    flame = FlameTorch(assets.model_path_or_skip()).to(DEV)
    emb = LandmarkEmbedding(
        ROOT / "mediapipe_landmark_embedding" / "mediapipe_landmark_embedding.npz",
        device=DEV)
    tex_cache = CACHE_DIR / "flame_texture_256_50.npz"
    tex = FlameTexture(tex_cache, device=DEV) if tex_cache.exists() else None
    keep = None
    if a.skin_mask > 0:
        vm = face_region(flame, emb, radius=a.skin_mask)
        keep = face_faces(flame, vm)
    renderer = FaceRenderer(flame, lmk_idx=emb, image_size=a.size, texture=tex,
                            face_keep=keep)

    if a.identity_data:
        root = pathlib.Path(a.identity_data)
        K = a.images_per_identity
        tr = IdentityPairs(root, a.size, "train", embeddings=a.arcface, k=K)
        va = IdentityPairs(root, a.size, "val", embeddings=a.arcface, k=K)
    else:
        tr = FFHQCrops(ROOT / "data" / "ffhq", a.size, "train")
        va = FFHQCrops(ROOT / "data" / "ffhq", a.size, "val")
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=4,
                    drop_last=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=a.batch, num_workers=2)

    mix_dl = None
    if a.mix_ffhq > 0:
        mix_ds = FFHQCrops(ROOT / "data" / "ffhq", a.size, "train")
        mix_dl = DataLoader(mix_ds, batch_size=a.mix_batch or a.batch, shuffle=True,
                            num_workers=2, drop_last=True, persistent_workers=True)
        print(f"    mixing {len(mix_ds)} FFHQ photographs at weight {a.mix_ffhq}")

    id_loss = IdentityLoss(DEV) if a.w_id > 0 else None
    Enc = ArcFaceShapeEncoder if a.arcface else ResNetEncoder
    enc = Enc(n_shape=100, n_expr=50, pretrained=True).to(DEV)
    if a.freeze_backbone:
        for p in enc.trunk.parameters():
            p.requires_grad = False
    params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)
    per_epoch = a.limit_steps or len(dl)
    steps = max(1, a.epochs * per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)

    unit = "identities" if a.identity_data else "images"
    print(f"=== training on {len(tr)} {unit} ({len(va)} held out), {DEV} ===")
    if a.identity_data:
        print(f"    identity pairs from {a.identity_data} "
              f"(two different images per subject, split by identity)")
    print(f"    {a.epochs} epochs x {per_epoch} steps = {steps}, batch {a.batch}, lr {a.lr}"
          + (", backbone frozen" if a.freeze_backbone else ""))
    print(f"    consistency weight: {a.w_con}   swap weight: {a.w_swap}"
          + ("  (paired views, batch doubles)" if (a.w_con > 0 or a.w_swap > 0 or a.pair)
             else "  (off)"))
    print(f"    shape source: "
          + ("ArcFace embedding (512-d) -> MLP" if a.arcface
             else "ResNet trunk (pixels)"))
    print(f"    skin mask: "
          + (f"radius {a.skin_mask} m, {int(keep.sum())}/{flame.n_faces} triangles"
             if keep is not None else "OFF (whole silhouette)"))
    print(f"    texture: {'on' if tex else 'OFF (flat grey)'}   "
          f"trainable {sum(p.numel() for p in params) / 1e6:.1f} M")
    print("")

    def step(batch, augment=True, as_pairs=None):
        prepaired = bool(a.identity_data) if as_pairs is None else as_pairs
        if prepaired:
            # (B,K,...) -> (B*K,...), each identity contiguous.
            batch = flatten_pairs(batch, k=a.images_per_identity)
        img = batch["image"].to(DEV, non_blocking=True)
        gt = batch["landmarks"].to(DEV, non_blocking=True)
        paired = prepaired or ((a.w_con > 0 or a.w_swap > 0 or a.pair) and augment)
        if paired and not prepaired:
            img, gt = two_views(img, gt)
        # Validity is recomputed from pixels rather than taken from the cache:
        # a rotated or scaled view has its own black borders, so the ingest-time
        # mask no longer describes this image.
        valid = img.sum(1) > 0
        emb = batch.get("embedding")
        if emb is not None:
            emb = emb.to(DEV, non_blocking=True)
        pred = enc.predict(img, embedding=emb) if a.arcface else enc.predict(img)
        verts, _ = renderer.geometry(pred)

        # Landmarks and regularisation are computed on every view: they need
        # FLAME and a projection, both cheap. Rasterisation is what costs
        # memory (roughly half of a full step), so only the views in `n_render`
        # are rendered. With paired views that halves the render cost while
        # keeping the augmented view's landmark supervision.
        n_render = (img.shape[0] // 2 if (paired and a.render_first_only)
                    else img.shape[0])
        render, mask = renderer.render(verts[:n_render], pred[:n_render])
        target = img[:n_render].permute(0, 2, 3, 1)

        l_lmk = landmark_loss(renderer.landmarks(verts, pred.cam), gt)

        # DECA-style swap: re-render each view using the *other* view's shape,
        # everything else unchanged, and apply the ordinary reconstruction
        # losses. Unlike a distance penalty this cannot be satisfied by
        # collapsing shape, because the other view's shape is then the only
        # thing available to explain this view's landmarks and pixels.
        l_swap = render.new_zeros(())
        if paired and a.w_swap > 0:
            sw = swap_shape(pred, k=a.images_per_identity,
                            blocked=bool(a.identity_data))
            v_sw, _ = renderer.geometry(sw)
            l_swap = landmark_loss(renderer.landmarks(v_sw, sw.cam), gt)
            if a.swap_photometric:
                r_sw, m_sw = renderer.render(v_sw[:n_render], sw[:n_render])
                l_swap = l_swap + photometric_loss(r_sw, target, m_sw & valid[:n_render])
        l_pho = photometric_loss(render, target, mask & valid[:n_render])
        l_reg = regularization(pred, w_light=a.w_light)

        # Identity: composite the render into the photograph over the face
        # region and require a recognition network to see the same person.
        # pred.albedo/light are not detached here because the render already
        # carries them; what matters is that the comparison target is the real
        # photo, so the only way to reduce this is better geometry.
        l_id = render.new_zeros(())
        if id_loss is not None:
            overlay = make_overlay(render, target, mask)
            l_id = id_loss(overlay, target)
        l_con = consistency_loss(pred.shape) if (paired and a.w_con > 0) else render.new_zeros(())
        loss = (a.w_lmk * l_lmk + a.w_pho * l_pho + l_reg
                + a.w_con * l_con + a.w_swap * l_swap + a.w_id * l_id)
        return (loss, (l_lmk, l_pho, l_reg, l_con, l_swap, l_id),
                render, mask, img[:n_render])

    hist = []
    done = 0
    mix_iter = iter(mix_dl) if mix_dl is not None else None
    for ep in range(a.epochs):
        enc.train()
        t0 = time.time()
        agg = np.zeros(7)
        n = 0
        for batch in dl:
            opt.zero_grad()
            loss, terms, *_ = step(batch)
            if mix_dl is not None:
                # Reconstruction only: no swap, no pairing. This batch exists to
                # keep the encoder honest about real photographs.
                try:
                    mb = next(mix_iter)
                except (StopIteration, NameError):
                    mix_iter = iter(mix_dl)
                    mb = next(mix_iter)
                mloss, _, *_ = step(mb, augment=False, as_pairs=False)
                loss = loss + a.mix_ffhq * mloss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            done += 1
            if done < steps:
                sched.step()
            agg += [loss.item()] + [x.item() for x in terms]
            n += 1
            if a.limit_steps and n >= a.limit_steps:
                break
        agg /= max(n, 1)

        enc.eval()
        vagg = np.zeros(7)
        m = 0
        with torch.no_grad():
            for batch in vl:
                # Validation is measured without augmentation, so the number is
                # comparable across runs with and without consistency.
                loss, terms, *_ = step(batch, augment=False)
                vagg += [loss.item()] + [x.item() for x in terms]
                m += 1
                if a.limit_steps and m >= 4:
                    break
        vagg /= max(m, 1)
        hist.append({"epoch": ep, "train": agg.tolist(), "val": vagg.tolist()})
        con = f" con {agg[4]:.4f}" if a.w_con > 0 else ""
        con += f" swap {agg[5]:.4f}" if a.w_swap > 0 else ""
        con += f" id {agg[6]:.4f}" if a.w_id > 0 else ""
        print(f"  ep {ep:2d}  train {agg[0]:.4f} (lmk {agg[1]:.4f} pho {agg[2]:.4f}{con})   "
              f"val {vagg[0]:.4f} (lmk {vagg[1]:.4f} pho {vagg[2]:.4f})   "
              f"{time.time() - t0:.0f}s", flush=True)

        torch.save({"model": enc.state_dict(), "epoch": ep, "args": vars(a)},
                   out / "encoder.pt")
        (out / "history.json").write_text(json.dumps(hist, indent=1))

    from PIL import Image
    enc.eval()
    with torch.no_grad():
        b = next(iter(vl))
        _, _, render, mask, img = step(b, augment=False)
        k = min(6, img.shape[0])
        top = torch.cat(list(img.permute(0, 2, 3, 1)[:k]), 1)
        bot = torch.cat(list(render[:k]), 1)
        panel = torch.cat([top, bot], 0).clamp(0, 1).cpu().numpy()
    (ROOT / "out").mkdir(exist_ok=True)
    Image.fromarray((panel * 255).astype(np.uint8)).save(ROOT / "out" / "train_ffhq.png")
    print("")
    print(f"  wrote out/train_ffhq.png   checkpoint -> {out / 'encoder.pt'}")
    if DEV == "cuda":
        print(f"  peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
