"""A parametric, differentiable eye texture — the eyes' own albedo.

Why this exists. The 50-component albedo PCA allocates its variance by pixel
AREA, and the eyes are 15% of the UV map carrying almost none of the basis
energy. So every reconstruction gets a population-average iris: measured, the
basis mean's eye region is a blurred brown blob with no white sclera at all,
because averaging over many eyes and gaze directions destroys the structure.
Iris colour is one of the strongest identity cues a face has, and the pipeline
was discarding it.

An eye is also far better conditioned than skin. Skin needs 50 PCA components;
an eye is almost fully described by two colours, because the geometry is fixed:

    a 12 mm iris on a 24 mm eyeball subtends asin(6/12) = 30 degrees
    a  4 mm pupil subtends asin(2/12) = 10 degrees

Those come from anatomy, not from data, so they are constants here and only the
COLOURS are predicted. That matters because the eye region of a 224px crop is
roughly 20x10 pixels: enough to recover hue, nowhere near enough to estimate a
radius. Predicting fewer things badly-observed is the whole point.

The geometry is derived from the MESH, not from the texture. Each eyeball is a
sphere of 546 vertices; the angle between a vertex's outward direction and the
eye's forward axis says exactly where it sits, with no projection convention to
guess at.
"""

import numpy as np
import torch

# Anatomy, in degrees from the forward axis. Not tunable parameters.
PUPIL_DEG = 10.0
IRIS_DEG = 30.0
LIMBUS_DEG = 34.0        # the dark ring where iris meets sclera

# Feather widths, so the boundaries are not aliased staircases at 27 px across.
SOFT_DEG = 3.0


def eye_theta_map(flame, vt, ft, resolution=256, eye_mask=None):
    """(H,W) float: angle in degrees from the eye's forward axis, -1 elsewhere.

    Rasterised per triangle rather than per texel. The eye discs are ~27 px
    across and carry 2,176 triangles between them, so a triangle covers about
    one texel and flat-filling each with its mean angle is effectively exact.
    """
    from PIL import Image, ImageDraw

    from ..geometry.facemask import eye_faces
    from ..geometry.rig import JOINT_NAMES

    if eye_mask is None:
        eye_mask = eye_faces(flame)
    eye_mask = eye_mask.cpu().numpy() if torch.is_tensor(eye_mask) else eye_mask

    with torch.no_grad():
        v, _ = flame(batch_size=1)
    v = v[0].cpu().numpy()
    W = flame.weights.cpu().numpy()

    # Per-vertex angle, measured about that vertex's OWN eyeball centre. The two
    # eyes are separate spheres, so a single shared centre would skew both.
    theta = np.full(len(v), -1.0, np.float32)
    for ji, name in enumerate(JOINT_NAMES):
        if "eye" not in name.lower():
            continue
        m = W[:, ji] > 0.5
        if not m.any():
            continue
        d = v[m] - v[m].mean(0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        # +z is the direction the face looks along, so it is also the direction
        # the eyeballs look along at rest.
        theta[m] = np.degrees(np.arccos(np.clip(d[:, 2], -1.0, 1.0)))

    img = Image.new("F", (resolution, resolution), -1.0)
    draw = ImageDraw.Draw(img)
    faces = flame.faces.cpu().numpy()
    for tri_uv, tri_v in zip(np.asarray(ft)[eye_mask], faces[eye_mask]):
        t = theta[tri_v]
        if (t < 0).any():
            continue
        pts = [(float(vt[i, 0] * resolution),
                float((1.0 - vt[i, 1]) * resolution)) for i in tri_uv]
        draw.polygon(pts, fill=float(t.mean()))
    return np.asarray(img, dtype=np.float32)


def _ramp(x, edge, width):
    """Smooth 0->1 crossing at `edge`. torch.sigmoid keeps it differentiable;
    a hard comparison would give the encoder no gradient to follow."""
    return torch.sigmoid((x - edge) / max(width, 1e-6))


def eye_texture(iris_rgb, sclera_rgb, theta, resolution=None):
    """(B,3,H,W) eye albedo from two predicted colours.

    iris_rgb, sclera_rgb  (B,3) in [0,1]
    theta                 (H,W) from eye_theta_map, -1 outside the eyes

    Composited outward: pupil, then iris, then a dark limbal ring, then sclera.
    Everything is a smooth sigmoid blend, so gradients reach the colours from
    every texel rather than only from the ones that happen to sit on a boundary.
    """
    B = iris_rgb.shape[0]
    dev, dt = iris_rgb.device, iris_rgb.dtype
    th = torch.as_tensor(theta, device=dev, dtype=dt)[None, None]     # (1,1,H,W)

    iris = iris_rgb.view(B, 3, 1, 1)
    sclera = sclera_rgb.view(B, 3, 1, 1)

    # The pupil is an aperture, not a pigment: it is the inside of the eye seen
    # through the lens, so it is near-black regardless of iris colour. A little
    # of the iris colour bleeds in, which is what real pupils look like.
    pupil = iris * 0.06

    out = pupil + (iris - pupil) * _ramp(th, PUPIL_DEG, SOFT_DEG)
    out = out + (sclera - out) * _ramp(th, LIMBUS_DEG, SOFT_DEG)

    # The limbal ring: a dark annulus at the iris edge. Present in every real
    # eye, and its absence is one reason CG irises look printed on.
    ring = torch.exp(-0.5 * ((th - IRIS_DEG) / 2.5) ** 2)
    out = out * (1.0 - 0.45 * ring)

    return out.expand(B, 3, th.shape[-2], th.shape[-1]).contiguous()


def default_eye_params(batch=1, device="cpu"):
    """Mid-brown iris, faintly warm sclera. The starting point the encoder's
    head is biased to, so an untrained model emits a plausible eye rather than
    a black one."""
    iris = torch.tensor([0.32, 0.22, 0.14], device=device).repeat(batch, 1)
    sclera = torch.tensor([0.88, 0.85, 0.82], device=device).repeat(batch, 1)
    return iris, sclera
