"""Look at what the encoder actually outputs, next to what it was given.

Two figures, because they answer different questions.

  reconstruction  input | textured render | shaded geometry | neutral identity
                  Does the output explain the photograph?

  identity        one subject, several photographs, each reduced to its neutral
                  shape under one fixed frontal camera. If identity were being
                  learned these would be the same face every time. The measured
                  between/within ratio of 0.42 says they will not be, and this
                  is that number made visible.

The neutral column is the important one: it zeroes expression and pose, so what
remains is purely what the encoder believes about who this person is.
"""

import argparse
import collections
import pathlib
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d import assets, now
from face3d.albedo import CACHE_DIR, FlameTexture
from face3d.detect import FaceDetector, crop_square
from face3d.encoder import ResNetEncoder
from face3d.flame_torch import FlameTorch
from face3d.params import FlameParams
from face3d.pipeline import FaceRenderer

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES = ROOT / "NoW_Dataset" / "final_release_version" / "iphone_pictures"
SIZE = 224


def label(arr, text):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(im) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs" / "base" / "encoder.pt"))
    ap.add_argument("--subjects", type=int, default=4)
    ap.add_argument("--per-subject", type=int, default=5)
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()

    ckpt = pathlib.Path(a.checkpoint)
    if not (ckpt.exists() and IMAGES.exists()):
        print("SKIP - needs a checkpoint and the NoW images")
        sys.exit(0)

    flame = FlameTorch(assets.model_path_or_skip()).to(DEV)
    tex = FlameTexture(CACHE_DIR / "flame_texture_256_50.npz", device=DEV)
    rend = FaceRenderer(flame, image_size=SIZE, texture=tex)
    plain = FaceRenderer(flame, image_size=SIZE)          # no texture: grey shading
    enc = ResNetEncoder(n_shape=100, n_expr=50, pretrained=False).to(DEV)
    enc.load_state_dict(torch.load(ckpt, map_location=DEV)["model"])
    enc.eval()

    # Group validation images by subject, preferring the categories MediaPipe
    # handles, so a detector failure does not masquerade as an encoder failure.
    by_subject = collections.defaultdict(list)
    for rel in now.image_list(ROOT, "validation"):
        by_subject[rel.split("/")[0]].append(rel)

    det = FaceDetector(blendshapes=False)

    def encode(rel):
        im = np.asarray(Image.open(IMAGES / rel).convert("RGB"))
        r = det.detect(im)
        if r is None:
            return None
        crop, _ = crop_square(im, r["norm"], size=SIZE)
        x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)[None]
        with torch.no_grad():
            p = enc.predict(x.float().to(DEV) / 255.0)
        return crop / 255.0, p

    # A fixed frontal camera and flat lighting for every neutral render, so the
    # only thing that can differ between them is the predicted identity.
    def neutral(p):
        n = FlameParams(
            shape=p.shape, expr=torch.zeros_like(p.expr), pose=torch.zeros_like(p.pose),
            cam=torch.tensor([[5.6, 0.0, 0.16]], device=DEV),
            light=p.light.new_zeros(1, 9, 3), albedo=p.albedo)
        # The SH DC basis is 0.282, so a coefficient of ~3 is needed for a
        # mid-grey surface; 0.95 renders almost black and hides the geometry
        # this figure exists to show.
        n.light[:, 0] = 3.2
        n.light[:, 2] = 1.1
        n.light[:, 3] = 0.8
        v, _ = plain.geometry(n)
        img, _ = plain.render(v, n)
        return img[0].cpu().numpy()

    subjects = sorted(by_subject)[: a.subjects]

    # --- figure 1: reconstruction quality -----------------------------------
    rows = []
    for s in subjects:
        for rel in by_subject[s]:
            got = encode(rel)
            if got is None:
                continue
            crop, p = got
            v, _ = rend.geometry(p)
            with torch.no_grad():
                textured, _ = rend.render(v, p)
                shaded, _ = plain.render(v, p)
            rows.append(np.concatenate([
                label(crop, "input"),
                label(textured[0].cpu().numpy(), "predicted (textured)"),
                label(shaded[0].cpu().numpy(), "geometry only"),
                label(neutral(p), "neutral identity")], axis=1))
            break
    if rows:
        Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)).save(
            OUT / f"predictions_{a.tag}.png")
        print(f"wrote predictions_{a.tag}.png  ({len(rows)} subjects)")

    # --- figure 2: is identity stable across photographs of one person? ------
    rows = []
    for s in subjects:
        tiles, n = [], 0
        for rel in by_subject[s]:
            got = encode(rel)
            if got is None:
                continue
            crop, p = got
            tiles.append(np.concatenate([label(crop, rel.split("/")[1][:14]),
                                         label(neutral(p), "-> identity")], axis=0))
            n += 1
            if n >= a.per_subject:
                break
        if tiles:
            rows.append(np.concatenate(tiles, axis=1))
    det.close()
    if rows:
        w = min(r.shape[1] for r in rows)
        Image.fromarray((np.concatenate([r[:, :w] for r in rows], 0) * 255)
                        .astype(np.uint8)).save(OUT / f"identity_{a.tag}.png")
        print(f"wrote identity_{a.tag}.png  ({len(rows)} subjects x {a.per_subject} photos)")


if __name__ == "__main__":
    main()
