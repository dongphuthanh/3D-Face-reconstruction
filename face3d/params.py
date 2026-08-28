"""The public parameter type — story B2.

This dataclass is the contract between the encoder and everything downstream.
Keeping it explicit and typed (rather than passing a bare 236-vector around) is
what lets DECA, SMIRK and MICA sit behind one interface: they disagree about how
many coefficients they emit, not about what the coefficients mean.
"""

from dataclasses import dataclass, fields, replace
from typing import Optional

import torch


@dataclass
class FlameParams:
    """Per-image FLAME coefficients. All tensors are batched, first dim B.

    shape  (B, n_shape)  identity, unitless PCA coefficients
    expr   (B, n_expr)   expression, same basis convention
    pose   (B, 15)       axis-angle for 5 joints: global, neck, jaw, L eye, R eye
    cam    (B, 3)        weak perspective: scale, tx, ty
    light  (B, 9, 3)     order-2 spherical harmonics, per RGB channel
    albedo (B, n_alb)    BFM albedo coefficients, or None while albedo is out of scope

    light and albedo exist only so a photometric loss can render something
    comparable to the input; they are not part of the exported asset.
    """

    shape: torch.Tensor
    expr: torch.Tensor
    pose: torch.Tensor
    cam: torch.Tensor
    light: torch.Tensor
    albedo: Optional[torch.Tensor] = None
    # (B,6): iris RGB then sclera RGB. The eyes get their own two colours
    # because the 50-component albedo basis allocates variance by pixel area
    # and leaves them a population-average iris. See face3d/eyes.py.
    eye: Optional[torch.Tensor] = None

    def __post_init__(self):
        b = self.shape.shape[0]
        for name in ("expr", "pose", "cam", "light"):
            t = getattr(self, name)
            if t.shape[0] != b:
                raise ValueError(f"batch mismatch: shape={b} but {name}={t.shape[0]}")
        if self.pose.shape[-1] != 15:
            raise ValueError(f"pose must be 15 axis-angle values, got {self.pose.shape[-1]}")
        if self.light.shape[-2:] != (9, 3):
            raise ValueError(f"light must be (B,9,3), got {tuple(self.light.shape)}")

    @property
    def batch_size(self) -> int:
        return self.shape.shape[0]

    @property
    def jaw(self) -> torch.Tensor:
        """Jaw rotation — joint 2. The only pose joint DECA-style encoders drive."""
        return self.pose[:, 6:9]

    def __getitem__(self, idx) -> "FlameParams":
        """Slice along the batch. Needed to render a subset of a batch -- e.g.
        paired-view training, where both views are encoded but only one is
        rasterised, since the render path dominates memory."""
        return self._map(lambda t: t[idx])

    def to(self, device) -> "FlameParams":
        return self._map(lambda t: t.to(device))

    def _map(self, fn) -> "FlameParams":
        """Apply fn to every tensor field, skipping None.

        Iterating the dataclass fields rather than naming them: the earlier
        version listed shape/expr/pose/cam/light/albedo by hand, so adding `eye`
        silently left it unsliced. Rendering a subset of a paired batch then hit
        a shape mismatch deep inside the texture compositor, a long way from the
        line that actually caused it.
        """
        return replace(self, **{
            f.name: fn(v)
            for f in fields(self)
            if (v := getattr(self, f.name)) is not None
        })

    def detach(self) -> "FlameParams":
        # Same reasoning as _map: naming the fields here would have left `eye`
        # attached, so a "detached" params object would still have carried
        # gradient into the eye colours.
        return self._map(lambda t: t.detach())

    def pad_to(self, n_shape: int, n_expr: int) -> "FlameParams":
        """Zero-extend to a full FLAME basis.

        Encoders predict a truncated basis (DECA: 100 shape, 50 expr) while FLAME
        expects 300/100. Doing this in one place stops every call site from
        reinventing the padding, and getting it wrong shifts every coefficient.
        """
        def pad(t, n):
            if t.shape[1] > n:
                raise ValueError(f"cannot truncate {t.shape[1]} coefficients to {n}")
            if t.shape[1] == n:
                return t
            return torch.cat([t, t.new_zeros(t.shape[0], n - t.shape[1])], dim=1)
        return replace(self, shape=pad(self.shape, n_shape), expr=pad(self.expr, n_expr))
