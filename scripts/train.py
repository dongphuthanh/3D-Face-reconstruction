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
from face3d.data import FFHQCrops
from face3d.encoder import ResNetEncoder
from face3d.facemask import face_faces, face_region
from face3d.flame_torch import FlameTorch
from face3d.landmarks import LandmarkEmbedding
from face3d.augment import consistency_loss, two_views
from face3d.losses import landmark_loss, photometric_loss, regularization
from face3d.pipeline import FaceRenderer

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-lmk", type=float, default=5.0)
    ap.add_argument("--w-pho", type=float, default=1.0)
    ap.add_argument("--w-con", type=float, default=0.0,
                    help="shape-consistency across two augmented views; "
                         "0 reproduces the original baseline")
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

    tr = FFHQCrops(ROOT / "data" / "ffhq", a.size, "train")
    va = FFHQCrops(ROOT / "data" / "ffhq", a.size, "val")
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=4,
                    drop_last=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=a.batch, num_workers=2)

    enc = ResNetEncoder(n_shape=100, n_expr=50, pretrained=True).to(DEV)
    if a.freeze_backbone:
        for p in enc.trunk.parameters():
            p.requires_grad = False
    params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)
    per_epoch = a.limit_steps or len(dl)
    steps = max(1, a.epochs * per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)

    print(f"=== training on {len(tr)} images ({len(va)} held out), {DEV} ===")
    print(f"    {a.epochs} epochs x {per_epoch} steps = {steps}, batch {a.batch}, lr {a.lr}"
          + (", backbone frozen" if a.freeze_backbone else ""))
    print(f"    consistency weight: {a.w_con}"
          + ("  (two augmented views, batch doubles)" if a.w_con > 0 else "  (off)"))
    print(f"    skin mask: "
          + (f"radius {a.skin_mask} m, {int(keep.sum())}/{flame.n_faces} triangles"
             if keep is not None else "OFF (whole silhouette)"))
    print(f"    texture: {'on' if tex else 'OFF (flat grey)'}   "
          f"trainable {sum(p.numel() for p in params) / 1e6:.1f} M")
    print("")

    def step(batch, augment=True):
        img = batch["image"].to(DEV, non_blocking=True)
        gt = batch["landmarks"].to(DEV, non_blocking=True)
        paired = a.w_con > 0 and augment
        if paired:
            img, gt = two_views(img, gt)
        # Validity is recomputed from pixels rather than taken from the cache:
        # a rotated or scaled view has its own black borders, so the ingest-time
        # mask no longer describes this image.
        valid = img.sum(1) > 0
        pred = enc.predict(img)
        verts, _ = renderer.geometry(pred)

        # Landmarks and regularisation are computed on every view: they need
        # FLAME and a projection, both cheap. Rasterisation is what costs
        # memory (roughly half of a full step), so only the views in `n_render`
        # are rendered. With paired views that halves the render cost while
        # keeping the augmented view's landmark supervision.
        n_render = img.shape[0] // 2 if (paired and a.render_first_only) else img.shape[0]
        render, mask = renderer.render(verts[:n_render], pred[:n_render])
        target = img[:n_render].permute(0, 2, 3, 1)

        l_lmk = landmark_loss(renderer.landmarks(verts, pred.cam), gt)
        l_pho = photometric_loss(render, target, mask & valid[:n_render])
        l_reg = regularization(pred)
        l_con = consistency_loss(pred.shape) if paired else render.new_zeros(())
        loss = a.w_lmk * l_lmk + a.w_pho * l_pho + l_reg + a.w_con * l_con
        return loss, (l_lmk, l_pho, l_reg, l_con), render, mask, img[:n_render]

    hist = []
    done = 0
    for ep in range(a.epochs):
        enc.train()
        t0 = time.time()
        agg = np.zeros(5)
        n = 0
        for batch in dl:
            opt.zero_grad()
            loss, terms, *_ = step(batch)
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
        vagg = np.zeros(5)
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
