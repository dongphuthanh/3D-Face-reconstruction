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

# The encoder is over-confident about identity: its shape direction correlates
# with the true face but its magnitude overshoots. Measured by sweeping a scale
# factor over the predicted shape and scoring each on NoW (0.00 is the FLAME
# mean face, the Bayes-optimal guess under no information):
#
#   model      0.00    0.25    0.50    0.75    1.00
#   id_swap   1.3554  1.3300  1.3485  1.4214  1.5296   (2k identities)
#   digi_all  1.3554  1.3294  1.3241  1.3602  1.4242   (10k identities)
#
# The optimum moved 0.25 -> 0.50 as identity data scaled 5x, i.e. twice as much
# of the prediction became usable: better calibration, not just a better score.
# Extrapolating, an optimum at 1.0 would mean no post-hoc scaling is needed.
#
# Re-measure with scripts/now_predict.py --shape-scale after any training
# change; this constant is an empirical patch over a calibration gap, not a
# property of the architecture.
SHAPE_CALIBRATION = 0.50


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
        self.n_out = n_shape + n_expr + 6 + 3 + 27 + n_albedo

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
        # no gradient at all on the first step. std=1e-3 keeps the initial
        # prediction within ~0.05 of the bias (ResNet-50 features are large enough that (still the mean face) while
        # letting the trunk train from iteration one.
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

    # ImageNet statistics; the trunk expects them whether or not it is pretrained
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def forward(self, image: torch.Tensor) -> FlameParams:
        m = image.new_tensor(self.MEAN).view(1, 3, 1, 1)
        s = image.new_tensor(self.STD).view(1, 3, 1, 1)
        code = self.head(self.trunk((image - m) / s))
        return self._split(code)

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

        # Expand the 6 predicted values into FLAME's 15-value pose vector:
        # global rotation and jaw are driven, neck and both eyes stay at rest.
        pose = code.new_zeros(B, 15)
        pose[:, 0:3] = rot6[:, 0:3]
        pose[:, 6:9] = rot6[:, 3:6]
        return FlameParams(shape=shape, expr=expr, pose=pose, cam=cam,
                           light=light.view(B, 9, 3), albedo=albedo)
