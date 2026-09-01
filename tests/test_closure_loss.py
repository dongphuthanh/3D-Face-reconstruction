"""The eye and lip pairs must actually be eyelids and lips.

EYE_PAIRS and LIP_PAIRS are row indices into our 105-point MediaPipe embedding,
derived by hand from MediaPipe's 0-467 numbering. A transcription slip there
does not crash anything -- it trains the model to match the distance between two
arbitrary points on the face, and shows up only as a loss that will not go down.
So check the geometry rather than trusting the numbers.

Note what a neutral FLAME face actually looks like: the eyes are OPEN and the
mouth is SHUT. So eyelid ordering is meaningful at rest, and lip ordering is not
-- the inner-lip pairs sit 0.005 NDC apart with an arbitrary sign. The first
version of this test asserted lip ordering at rest and failed on indices that
were correct. Lips are checked with the jaw open instead.
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d import assets
from face3d.geometry.flame_torch import FlameTorch
from face3d.geometry.landmarks import LandmarkEmbedding
from face3d.learn.losses import EYE_PAIRS, LIP_PAIRS, _pair_distance, closure_loss
from face3d.geometry.params import FlameParams
from face3d.render.pipeline import FaceRenderer

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def landmarks_for(renderer, jaw=0.0):
    z = lambda *s: torch.zeros(*s, device=DEV)
    pose = z(1, 15)
    pose[0, 6] = jaw     # FLAME pose: global(0-2) neck(3-5) jaw(6-8) eyes(9-14)
    p = FlameParams(shape=z(1, 100), expr=z(1, 50), pose=pose, albedo=z(1, 50),
                    light=z(1, 9, 3) + 0.5,
                    cam=torch.tensor([[7.0, 0.0, 0.0]], device=DEV))
    verts, _ = renderer.geometry(p)
    return renderer.landmarks(verts, p.cam)


def main():
    emb_path = ROOT / "mediapipe_landmark_embedding" / "mediapipe_landmark_embedding.npz"
    if not emb_path.exists():
        print("SKIP - no mediapipe landmark embedding")
        return 0
    flame = FlameTorch(assets.model_path_or_skip()).to(DEV)
    renderer = FaceRenderer(flame, lmk_idx=LandmarkEmbedding(emb_path, device=DEV),
                            image_size=224)

    neutral = landmarks_for(renderer)
    open_jaw = landmarks_for(renderer, jaw=0.3)

    print("vertical ordering (y-up; lips judged with the jaw open):")
    for name, pairs, lmk in (("eye", EYE_PAIRS, neutral),
                             ("lip", LIP_PAIRS, open_jaw)):
        for up, lo in pairs:
            dy = (lmk[0, up, 1] - lmk[0, lo, 1]).item()
            print(f"  {name} {up:3d} over {lo:3d}   dy {dy:+.5f}")
            assert dy > 0, f"{name} pair ({up},{lo}) is inverted -- wrong indices"

    d_lip_n = _pair_distance(neutral, LIP_PAIRS)
    d_lip_o = _pair_distance(open_jaw, LIP_PAIRS)
    d_eye_n = _pair_distance(neutral, EYE_PAIRS)
    d_eye_o = _pair_distance(open_jaw, EYE_PAIRS)
    eye_change = (d_eye_o / d_eye_n - 1).abs().max().item()
    print("")
    print(f"jaw open 0.3 rad:  lip {d_lip_n.mean():.5f} -> {d_lip_o.mean():.5f}"
          f"   eye change {eye_change * 100:.2f}%")

    # Absolute thresholds, not ratios: the shut-mouth distance is near
    # degenerate, so any ratio measured against it is inflated and proves little.
    assert d_lip_n.max() < 0.02, "neutral mouth is not shut -- wrong lip indices"
    assert d_lip_o.min() > 0.10, "opening the jaw did not open the lips"
    assert eye_change < 0.05, f"opening the jaw moved the eyes by {eye_change:.3f}"

    assert closure_loss(neutral, neutral, EYE_PAIRS).item() == 0.0
    assert closure_loss(neutral, open_jaw, LIP_PAIRS).item() > 0

    # Must survive a degenerate pair: a shut mouth puts both landmarks at the
    # same point, where sqrt has infinite gradient.
    flat = neutral.clone()
    for up, lo in LIP_PAIRS:
        flat[:, up] = flat[:, lo]
    flat.requires_grad_(True)
    closure_loss(flat, neutral, LIP_PAIRS).backward()
    assert torch.isfinite(flat.grad).all(), "NaN gradient on a fully closed mouth"

    print("")
    print("OK: pairs are anatomically correct and the loss is finite when shut")
    return 0


if __name__ == "__main__":
    sys.exit(main())
