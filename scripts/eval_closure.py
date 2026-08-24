"""Eyelid and lip closure error on held-out identities.

NoW cannot measure this. Its protocol scores a NEUTRAL mesh -- now_predict.py
--neutral zeroes expression and pose before writing, because the ground truth is
one neutral scan per subject and predicting expression into it costs 0.2 mm. Eye
and lip closure is pure expression, so it is discarded before scoring. A NoW
number is not evidence about those terms in either direction, and reading one as
if it were is how a term that works gets thrown away.

What they are accountable to is direct: on held-out identities, how far the
predicted eyelid and lip gaps sit from the detected ones. Reported as a fraction
of the true gap, since absolute NDC distances are not comparable across faces at
different scales.

DigiFace held-out, 800 images over 5,465 identities:

    model       eye err   lip err
    deca_conf     10.6%     16.5%
    deca_id       15.8%     24.4%
    deca_full      6.2%     11.9%
    deca_jit       6.8%     12.1%

The identity loss made closure noticeably WORSE. It rewards committing to a
distinctive face, and a distinctive face is apparently paid for partly in eyelid
and lip geometry that nothing else was watching. deca_full recovers that and
more than halves what remains. This is the whole reason the script exists: those
models sit within 0.02 mm of each other on NoW and hide a 4x spread in
expression fidelity.

ALWAYS SCORE A MODEL ON ITS OWN DOMAIN, or state that you did not. The held-out
corpus is a choice, and it decides the answer:

    held-out set          deca_full      deca_celswap
    CelebA, real photos   6.2 / 13.4%    5.2 / 12.2%
    DigiFace, synthetic   6.2 / 11.8%    7.5 / 16.2%

Each model wins on the corpus it trained on, and the gap is large enough to
invert the conclusion. Reading only the DigiFace row says deca_celswap loses
closure badly; that row is out-of-domain for it and measures domain transfer,
not closure. On real photographs -- which is what this project reconstructs --
deca_celswap is the better model on both eyes and lips.

Worth noting the asymmetry: deca_full degrades gently off-domain (6.2/11.8 ->
6.2/13.4) while deca_celswap degrades sharply (5.2/12.2 -> 7.5/16.2). DigiFace's
synthetic renders apparently teach a more transferable eyelid and lip model even
though they are worse in absolute terms on real faces.

--min-images MUST match training. The val split is a permutation of identities
surviving that filter, so a different value reshuffles it and quietly scores
faces the model trained on. CelebA needs --min-images 4 and
--cache landmarks_224_swap.npz.
"""
import argparse, pathlib, sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face3d import assets
from face3d.data import IdentityPairs, flatten_pairs
from face3d.encoder import ArcFaceShapeEncoder, ResNetEncoder
from face3d.flame_torch import FlameTorch
from face3d.landmarks import LandmarkEmbedding
from face3d.losses import EYE_PAIRS, LIP_PAIRS, _pair_distance
from face3d.pipeline import FaceRenderer

ap = argparse.ArgumentParser()
ap.add_argument("arms", nargs="*")
ap.add_argument("--data", default="digiface",
                help="corpus whose held-out identities are scored")
ap.add_argument("--cache", default="",
                help="landmark cache inside --data; CelebA needs "
                     "landmarks_224_swap.npz")
ap.add_argument("--min-images", type=int, default=2,
                help="MUST match the value the model was trained with. The val "
                     "split is a permutation of the identities that survive "
                     "this filter, so a different value reshuffles it and "
                     "scores faces the model actually trained on")
a = ap.parse_args()

DEV = "cuda"
flame = FlameTorch(assets.model_path_or_skip()).to(DEV)
emb = LandmarkEmbedding(ROOT / "mediapipe_landmark_embedding" /
                        "mediapipe_landmark_embedding.npz", device=DEV)
renderer = FaceRenderer(flame, lmk_idx=emb, image_size=224)

ds = IdentityPairs(ROOT / "data" / a.data, size=224, split="val", k=2,
                   cache_name=a.cache or None, min_images=a.min_images)
loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

print(f"{len(ds)} held-out identities from {a.data}"
      f"{' (' + a.cache + ')' if a.cache else ''}, min_images={a.min_images}")
print(f"{'model':12s} {'eye err':>10s} {'lip err':>10s} {'eye rel':>9s} {'lip rel':>9s}")
# Arms come from argv so a new run can be measured without editing this file.
# The hardcoded tuple silently omitted deca_jit: the table printed three rows,
# looked complete, and said nothing about the model actually under test.
ARMS = a.arms or ["deca_conf", "deca_id", "deca_full", "deca_jit"]
for arm in ARMS:
    ck = ROOT / "runs" / arm / "encoder.pt"
    if not ck.exists():
        print(f"{arm:12s} missing"); continue
    blob = torch.load(ck, map_location=DEV)
    Enc = ArcFaceShapeEncoder if blob.get("args", {}).get("arcface") else ResNetEncoder
    enc = Enc().to(DEV); enc.load_state_dict(blob["model"]); enc.eval()

    e_err = l_err = e_gt = l_gt = n = 0.0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= 25:
                break
            b = flatten_pairs(batch, k=2)
            img = b["image"].to(DEV)
            gt = b["landmarks"].to(DEV)
            p = enc.predict(img)
            verts, _ = renderer.geometry(p)
            proj = renderer.landmarks(verts, p.cam)
            for pairs, acc in ((EYE_PAIRS, "e"), (LIP_PAIRS, "l")):
                d_p = _pair_distance(proj, pairs)
                d_g = _pair_distance(gt, pairs)
                if acc == "e":
                    e_err += (d_p - d_g).abs().sum().item(); e_gt += d_g.sum().item()
                else:
                    l_err += (d_p - d_g).abs().sum().item(); l_gt += d_g.sum().item()
            n += img.shape[0]
    ne, nl = n * len(EYE_PAIRS), n * len(LIP_PAIRS)
    print(f"{arm:12s} {e_err/ne:10.5f} {l_err/nl:10.5f} "
          f"{100*e_err/e_gt:8.1f}% {100*l_err/l_gt:8.1f}%   (n={int(n)})")
