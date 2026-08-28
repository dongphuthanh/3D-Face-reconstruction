"""The Encoder abstraction — stories B1/B2, and Decision 3.

`Encoder.predict(image) -> FlameParams` is the escape hatch that makes the ML
phase safe to fail: DECA, SMIRK and MICA all satisfy it, so swapping the encoder
is a one-line change and never touches the asset pipeline. It is also what makes
the licence strategy executable — replacing a DECA-derived encoder with one we
trained ourselves is, structurally, the same swap.
"""

from dataclasses import replace
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torchvision

from .params import FlameParams

# The encoder overshoots on identity: its shape direction correlates with the
# true face but its magnitude is too large. Measured by sweeping a scale over
# the predicted shape and scoring each on NoW (0.00 is the FLAME mean face, the
# Bayes-optimal guess given no information):
#
#   model      0.00   0.10   0.20   0.25   0.30   0.40   0.50   0.60   0.80   1.00
#   id_swap   1.3554    -      -   1.3300   -      -   1.3485   -      -    1.5296
#   digi_all  1.3554    -      -   1.3294   -      -   1.3241   -      -    1.4242
#   digi_full 1.3554    -      -   1.3224   -      -   1.3109   -      -    1.3978
#   deca_conf 1.3554    -      -      -     -   1.2978   -   1.2923 1.3079 1.3499
#   deca_id   1.3554 1.3180 1.2801   -   1.2566 1.2470   -   1.2704 1.3482 1.4834
#   deca_full 1.3554      - 1.2897      - 1.2710 1.2679 1.2777 1.3002      -      -
#   deca_jit  1.3554      - 1.2842      - 1.2654 1.2654 1.2772 1.3032      -      -
#   deca_cel  1.3554      - 1.2995      - 1.2870 1.2910 1.3091 1.3426      -      -
#   deca_csw  1.3554      - 1.2690      - 1.2457 1.2408 1.2540 1.2851 1.3905 1.5485
#   deca_jnt  1.3554      - 1.2878      - 1.2697 1.2688 1.2815 1.3060      -      -
#
# The optimum ran 0.25 -> 0.50 -> 0.60 while data and loss weighting improved,
# and it was tempting to read that as progress toward needing no scaling at
# all. The identity loss reversed it to 0.40 and improved NoW at the same time,
# so that reading was wrong. What actually moves the optimum is how large a
# deviation the encoder commits to: deca_id's between-subject shape spread is
# 4.447 against deca_conf's 2.392, and a bolder prediction needs more
# shrinkage, not less.
#
# The two metrics genuinely pull apart here. NoW measures surface distance, so
# it rewards hedging toward the mean face; the identity ratio measures
# discriminability, so it rewards committing. deca_id is better on both, but
# its NoW curve is four times steeper across the grid (0.24 mm against 0.06),
# which is what a bolder model looks like -- more to gain and more to lose from
# getting this scalar wrong.
#
# deca_jit adds DECA's randomised crop scale to deca_full and changes nothing.
# NoW 1.2679 -> 1.2654, closure 6.2/11.9% -> 6.8/12.1%, identity ratio
# 1.15 -> 1.18: two metrics marginally better, one marginally worse, every
# difference far below what this setup can resolve. Seed variance has never
# been measured here, so the smallest defensible claim is that crop jitter has
# no effect large enough to detect -- not that it helps by 0.0025 mm. That was
# the last gap between this pipeline and DECA's, which makes the null result
# the useful part: the remaining distance to their number is not explained by
# a missing augmentation.
#
# deca_cel swaps the mix corpus FFHQ -> CelebA and is WORSE: 1.2870 against
# deca_full's 1.2679, identity ratio unmoved at 1.15, between-subject spread
# 4.466 -> 4.567. The prediction was that CelebA's identity diversity -- 0.98
# between-subject facenet distance against DigiFace's 0.76 -- would widen the
# predicted shape spread. It did not, for a structural reason that should have
# been checked first: diversity can only widen shape spread through a loss that
# CONTRASTS identities, and the only such loss is the swap, which runs solely
# on DigiFace. Mix batches feed the photometric and identity terms, both
# per-image. Changing the mix corpus could not have produced the effect.
#
# What it did change is sharpness, for the worse. FFHQ crops downsample from
# 1024px originals; CelebA-aligned is 178x218, so a 1.6 crop upsamples a ~110px
# face to fill 224. Measured Laplacian variance 366 against 232, a 1.58x drop,
# and the photometric loss is the main consumer of mix batches. FFHQ's 18k
# images repeated ~35x beat CelebA's 145k repeated ~3x, so corpus size was
# never the constraint either.
#
# deca_csw is the same corpus in the position that can actually use it: CelebA
# driving the SWAP loss, split by age and weight so an identity is not asked to
# hold shape across them, at matched steps (36 epochs x 1,069 = 38,484 against
# deca_full's 3 x 12,979). 1.2408 mm, the best measured -- 0.027 below
# deca_full and 0.006 below deca_id, the previous best.
#
# The identity-diversity prediction is half confirmed. Between-subject shape
# spread reached 4.842, the widest measured (deca_full 4.466), so real
# identities did widen what the encoder commits to. But within-subject spread
# grew in step, 3.870 -> 4.250, so the RATIO sat at 1.14 and did not improve.
# Wider and looser together, which is what a corpus whose identities are real
# but photographed across years should produce.
#
# Closure looked worse, 6.2/11.8% -> 7.5/16.2%, and that reading was wrong: it
# scored DigiFace held-out identities, in-domain for deca_full and out-of-domain
# for deca_csw, which never saw a DigiFace face. Scored on each model's own
# corpus the conclusion inverts:
#
#     held-out set          deca_full      deca_csw
#     CelebA, real photos   6.2 / 13.4%    5.2 / 12.2%
#     DigiFace, synthetic   6.2 / 11.8%    7.5 / 16.2%
#
# On real photographs -- what this project actually reconstructs -- deca_csw is
# better on both eyes and lips. It wins NoW, which is real scans of real people,
# and closure on real faces; deca_full wins only on synthetic faces. deca_csw is
# the model to ship.
#
# The asymmetry is still worth knowing: deca_full degrades gently off-domain
# (6.2/11.8 -> 6.2/13.4) where deca_csw degrades sharply (5.2/12.2 -> 7.5/16.2),
# so DigiFace teaches a more transferable eyelid and lip model even though it is
# worse in absolute terms on real faces.
#
# deca_jnt trains on both, each at batch 8 for 38,937 steps -- the exposure each
# had in its own run. The two metrics then disagree, and the disagreement is the
# finding:
#
#     model      NoW    closure CelebA   closure DigiFace   ratio
#     deca_full  1.2679   6.2 / 13.4%      6.2 / 11.8%      1.15
#     deca_csw   1.2408   5.2 / 12.2%      7.5 / 16.2%      1.14
#     deca_jnt   1.2688   5.0 / 11.1%      6.3 / 11.9%      1.16
#
# Joint training wins closure on BOTH domains -- best on real faces, matching
# deca_full on synthetic, so the transferability deca_csw lost is fully
# recovered. But its NoW lands at 1.2688, back at deca_full's level and 0.028
# above deca_csw. Adding DigiFace back bought expression fidelity and gave up
# the identity-shape gain that CelebA alone produced. The spreads say the same:
# between-subject 4.638 sits between deca_full's 4.466 and deca_csw's 4.842.
#
# So there is no single winner. NoW scores a NEUTRAL mesh, i.e. identity shape;
# closure scores expression. deca_csw is the better identity model and deca_jnt
# the better expression model. For a rigged head the neutral mesh is the base
# and the blendshapes carry expression, so this is a real product decision, not
# a metric artefact. Seed variance is still unmeasured, which is what would say
# whether 0.028 mm is worth choosing on at all.
#
# 36 passes over 8,557 subjects did not overfit: val flattened at 0.2106,
# 0.2108, 0.2108 across the last three epochs and never rose.
#
# HOW CLOSE IS THIS TO JUST EMITTING THE MEAN FACE? Closer than the headline
# numbers suggest, and the honest framing belongs here rather than in a commit
# message nobody re-reads.
#
# deca_csw's raw prediction, alpha 1.00, scores 1.5485 mm. The mean face scores
# 1.3554. The encoder's actual unshrunk output is WORSE THAN IGNORING THE PHOTO.
# Shrinking to 0.40 is not a calibration nicety, it is the only reason the model
# beats a constant, and even then by 0.115 mm.
#
# Measured over 200 FFHQ faces, neutral mesh, per-vertex:
#
#     raw prediction (a=1.0)   5.45 mm from mean face   4.63 mm between people
#     as shipped     (a=0.4)   2.18 mm from mean face   1.79 mm between people
#
# So two different people's shipped meshes differ by 1.79 mm on average while
# the error against ground truth is 1.24 mm. Signal is only ~1.4x the noise.
# Rendered side by side (out/mean_face_check.png), a toddler, a boy, an elderly
# woman and an adult man produce visibly near-identical neutral geometry.
#
# The fair context, which is not an excuse but is real. On NoW's NON-METRICAL
# protocol -- the one run_docker_eval runs, and the one every number in this
# file is on -- the mean face scores 1.3554, DECA 1.09, and MICA 0.98, the best
# published. So the entire achievable band is ~0.375 mm and this pipeline has
# taken ~31% of it, or ~43% of the way to DECA. Monocular identity shape is
# genuinely hard and every published number sits near the constant baseline.
#
# Do not mix in MICA's headline 1.08: that is the METRICAL leaderboard, a
# different and harder protocol that forbids scale optimisation.
#
# What this does NOT indict: expression (closure 5.0/11.1% on real faces),
# pose and camera (the overlay composites seamlessly), and albedo (skin tone is
# captured). Those parts work. It is specifically IDENTITY SHAPE that is weak,
# and no amount of corpus or augmentation work in this file has moved it much.
#
# It is fit on NoW validation, the same set reported on, so treat it as a
# calibration constant rather than as evidence. Re-measure with
# scripts/now_predict.py --shape-scale after any training change; it is
# specific to a checkpoint and carrying an old value to a new one silently
# mis-scales every exported face.
# Sclera is anatomy, not identity: it yellows with age and reddens when
# irritated, but nobody has brown whites. Predicting it freely let the
# photometric loss drive it to a mid-brown within one epoch, because at 20x10
# pixels the eye region is mostly eyelid and lash, so a darker eyeball lowers
# pixel error while ceasing to look like an eye. Iris stays free -- that is the
# identity signal the generator exists to capture.
SCLERA_BASE = torch.tensor([0.88, 0.85, 0.82])
SCLERA_RANGE = 0.09

SHAPE_CALIBRATION = 0.40   # deca_id, deca_full, deca_jit; was 0.60 for deca_conf


@runtime_checkable
class Encoder(Protocol):
    """Anything that turns a batch of face crops into FLAME coefficients."""

    n_shape: int
    n_expr: int

    def predict(self, image: torch.Tensor) -> FlameParams:
        """image: (B,3,H,W) float in [0,1], face-cropped. Returns FlameParams."""
        ...


class ResNetEncoder(nn.Module, Encoder):
    """DECA-shaped regressor: a ResNet trunk and one linear head.

    Default split is DECA's 236 = 100 shape + 50 expr + 6 pose + 3 cam + 27 light
    + 50 albedo, so a DECA checkpoint can be adapted without reshaping anything.
    Only global rotation and jaw are predicted; neck and eye joints stay at rest,
    which is what the published models do.
    """

    def __init__(self, n_shape=100, n_expr=50, n_albedo=50, arch="resnet50",
                 pretrained=False):
        super().__init__()
        self.n_shape, self.n_expr, self.n_albedo = n_shape, n_expr, n_albedo
        # +6 at the end: iris RGB and sclera RGB, driving face3d/eyes.py
        self.n_out = n_shape + n_expr + 6 + 3 + 27 + n_albedo + 6

        weights = "DEFAULT" if pretrained else None      # pretrained triggers a download
        trunk = getattr(torchvision.models, arch)(weights=weights)
        self.feat_dim = trunk.fc.in_features
        trunk.fc = nn.Identity()
        self.trunk = trunk
        self.head = nn.Linear(self.feat_dim, self.n_out)

        # Start at the mean face: zero coefficients, unit camera scale, flat light.
        # Random init here means the first forward pass renders a mesh somewhere
        # off-screen and the photometric loss has no gradient to work with.
        # Small-gain rather than exactly zero: a zero weight matrix makes
        # dL/d(trunk) = dL/d(out) @ W identically zero, so the backbone receives
        # no gradient at all on the first step. std=3e-4 keeps the initial
        # prediction within ~0.05 of the bias (still the mean face) while
        # letting the trunk train from iteration one. The gain has to be this
        # small because ResNet-50 pools 2048 features: at std=1e-3 the summed
        # contribution is large enough to move the starting mesh off the mean.
        nn.init.normal_(self.head.weight, std=3e-4)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            c = n_shape + n_expr + 6
            # A FLAME head is ~0.32 m tall in model units, so a weak-perspective
            # scale near 5.6 fills the viewport; ty lifts the mesh centroid to
            # the middle of frame. Initialising at 1.0 would render a small head
            # in the corner and start the photometric loss with almost no signal.
            self.head.bias[c + 0] = 5.6                          # cam scale
            self.head.bias[c + 2] = 0.16                         # cam ty
            self.head.bias[c + 3 : c + 6] = 0.7                  # ambient SH, RGB
            # Eye colours start at a mid-brown iris and a faintly warm sclera,
            # pre-sigmoid. Starting at 0 would give a black iris and a mid-grey
            # sclera, and the first renders would look eyeless.
            e = self.n_out - 6
            # Iris: pre-sigmoid values giving a mid-brown [0.32, 0.22, 0.14].
            self.head.bias[e + 0 : e + 3] = torch.tensor([-0.75, -1.27, -1.82])
            # Sclera: zero, because tanh(0) = 0 puts it exactly on SCLERA_BASE.
            self.head.bias[e + 3 : e + 6] = 0.0

    # ImageNet statistics; the trunk expects them whether or not it is pretrained
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def forward(self, image: torch.Tensor) -> FlameParams:
        m = image.new_tensor(self.MEAN).view(1, 3, 1, 1)
        s = image.new_tensor(self.STD).view(1, 3, 1, 1)
        code = self.head(self.trunk((image - m) / s))
        return self._split(code)

    def load_state_dict(self, state_dict, strict=True):
        """Accept checkpoints that predate the eye outputs.

        Adding iris and sclera colour grew the head from 236 to 242, which makes
        every earlier checkpoint fail a strict load. Rather than orphan them,
        copy what the checkpoint has and leave the new rows at their initialised
        values -- so an old model still runs and simply emits the default
        mid-brown eye. Silently dropping a size mismatch would be worse than
        either option, so this only ever pads, never truncates.
        """
        w = state_dict.get("head.weight")
        if w is not None and w.shape[0] != self.head.weight.shape[0]:
            if w.shape[0] > self.head.weight.shape[0]:
                raise ValueError(
                    f"checkpoint head is {w.shape[0]} wide but this encoder "
                    f"emits {self.head.weight.shape[0]}; refusing to truncate")
            state_dict = dict(state_dict)
            n = w.shape[0]
            nw = self.head.weight.detach().clone(); nw[:n] = w
            nb = self.head.bias.detach().clone(); nb[:n] = state_dict["head.bias"]
            state_dict["head.weight"], state_dict["head.bias"] = nw, nb
            print(f"    note: checkpoint predates the eye outputs "
                  f"({n} -> {self.head.weight.shape[0]}); using default eye colour")
        return super().load_state_dict(state_dict, strict=strict)

    def predict(self, image: torch.Tensor, calibrate: bool = False) -> FlameParams:
        """calibrate=True applies SHAPE_CALIBRATION. Off during training, since
        the loss should see the raw prediction; on for inference and export."""
        p = self(image)
        if calibrate:
            p = replace(p, shape=p.shape * SHAPE_CALIBRATION)
        return p

    def _split(self, code: torch.Tensor) -> FlameParams:
        B = code.shape[0]
        i = 0
        def take(n):
            nonlocal i
            out = code[:, i:i + n]; i += n
            return out

        shape, expr = take(self.n_shape), take(self.n_expr)
        rot6 = take(6)
        cam, light, albedo = take(3), take(27), take(self.n_albedo)
        # Sigmoid because these are colours: an unbounded linear output would
        # let the encoder ask for negative or super-white pigment, which the
        # photometric loss cannot punish once it clips.
        #
        # IRIS is free across the full range -- eye colour is exactly the
        # identity signal this is here to capture, and it varies enormously.
        #
        # SCLERA is clamped to a narrow band around white, and that is not
        # timidity. Left free it COLLAPSED: measured after one epoch it had
        # drifted from [0.88 0.85 0.82] to [0.47 0.34 0.29], a mid-brown, i.e.
        # the model painting the whites of the eyes skin-coloured. The cause is
        # the loss, not the parameterisation -- at 20x10 pixels the eye region
        # is mostly eyelid, lash and shadow, so a darker eyeball genuinely
        # lowers pixel error even though it stops looking like an eye.
        #
        # Anatomy is the right prior here. Human sclera really is near-constant:
        # it yellows with age and reddens when irritated, but nobody has brown
        # whites. +/-0.09 covers the real variation and forecloses the collapse.
        raw = take(6)
        iris = torch.sigmoid(raw[:, :3])
        sclera = SCLERA_BASE.to(raw.device) + SCLERA_RANGE * torch.tanh(raw[:, 3:])
        eye = torch.cat([iris, sclera.clamp(0.0, 1.0)], dim=1)

        # Expand the 6 predicted values into FLAME's 15-value pose vector:
        # global rotation and jaw are driven, neck and both eyes stay at rest.
        pose = code.new_zeros(B, 15)
        pose[:, 0:3] = rot6[:, 0:3]
        pose[:, 6:9] = rot6[:, 3:6]
        return FlameParams(shape=shape, expr=expr, pose=pose, cam=cam,
                           light=light.view(B, 9, 3), albedo=albedo, eye=eye)


class ArcFaceShapeEncoder(ResNetEncoder):
    """MICA-style: identity comes from a face-recognition embedding, not pixels.

    Five of the six output groups already work -- pose, expression, camera,
    lighting and albedo all transfer visibly -- so the ResNet trunk keeps them.
    Only the shape head is replaced, because that is the one measured to be
    inert: between/within subject spread of 0.76 on NoW, meaning shape varies
    almost as much across photographs of one person as across different people.

    An ArcFace embedding of the same images scores 1.56 on that measure. It was
    trained on millions of faces to encode identity while discarding pose and
    lighting, which is precisely what our shape head kept absorbing. Mapping
    512-d -> shape is a small, low-dimensional problem; extracting identity from
    a 1-2 px signal in pixels is not.

    MICA uses a single linear layer here and trains it against registered 3D
    scans. Without scans the mapping is learned through the same self-supervised
    losses as before, so an MLP with one hidden layer is used instead -- the
    supervision is weaker and a little more capacity is warranted.
    """

    def __init__(self, n_shape=100, n_expr=50, n_albedo=50, arch="resnet50",
                 pretrained=False, embed_dim=512, hidden=512):
        super().__init__(n_shape=n_shape, n_expr=n_expr, n_albedo=n_albedo,
                         arch=arch, pretrained=pretrained)
        self.embed_dim = embed_dim
        self.shape_head = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_shape))
        # Start at the mean face, as the ResNet head does: a random init here
        # renders a distorted head on step one and the photometric term has
        # nothing useful to say about it.
        nn.init.normal_(self.shape_head[0].weight, std=1e-2)
        nn.init.zeros_(self.shape_head[0].bias)
        nn.init.normal_(self.shape_head[2].weight, std=1e-3)
        nn.init.zeros_(self.shape_head[2].bias)

    def predict(self, image, calibrate: bool = False, embedding=None):
        p = self(image)
        if embedding is not None:
            # ArcFace embeddings are unit-normalised; cached as float16.
            p = replace(p, shape=self.shape_head(embedding.to(p.shape.dtype)))
        if calibrate:
            p = replace(p, shape=p.shape * SHAPE_CALIBRATION)
        return p
