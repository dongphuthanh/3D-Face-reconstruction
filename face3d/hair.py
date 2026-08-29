"""Hair, as a scalp shell fitted to the photograph's hair silhouette.

FLAME is a bald skull. It has no hair geometry and no basis that could grow
any, so every head this project has exported reads as a mannequin from the
eyebrows up. This does not add a hair MODEL -- it inflates the scalp FLAME
already has, by an amount measured from the photograph, and lets the texture
projection paint the person's actual hair onto the result.

Be clear about what this is and is not. It is a cap that follows the outline of
the hair as seen from one view. It cannot do a parting, a curl, a strand, or
anything that leaves the skull -- a ponytail, a fringe hanging over an eye,
long hair falling past the shoulders. For short and medium hair it turns a bald
mannequin into something recognisable; for long hair it produces a helmet whose
silhouette is right from the front and wrong everywhere else.

Doing better means real hair geometry, which means either an authored asset
library (what shipping avatar products use) or strand reconstruction
(NeuralHDHair, GaussianHaircut -- research grade, multi-view, and far past the
compute here). Neither is a bigger version of this; they are different things.

The three steps:

  segment   MediaPipe's selfie multiclass model marks hair pixels
  profile   compare the hair outline to the head outline, per angle around the
            head, to get a thickness
  inflate   push the scalp out along its normals by that thickness, tapered to
            nothing at the hairline so it does not detach from the forehead

Colour is not handled here at all. The shell is textured by face3d/project.py
like the rest of the head: once the geometry is in roughly the right place, the
photograph's own hair pixels land on it.
"""

import pathlib

import numpy as np
import torch

from .facemask import eye_faces, face_region
from .render import vertex_normals
from .rig import JOINT_NAMES

MODEL = (pathlib.Path(__file__).resolve().parents[1] / "models" /
         "selfie_multiclass_256x256.tflite")

# Category ids in MediaPipe's selfie multiclass segmenter.
BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHER = range(6)

# Hair thicker than this is not a cap any more, it is a hairstyle, and inflating
# the scalp that far gives a balloon rather than a head. Long hair hanging BESIDE
# the face is what pushes the measurement up here, and the shell cannot
# represent that however far we push it. 4 cm.
MAX_THICKNESS = 0.040

# Below this the person is bald, or wearing a hat, or the segmenter found a few
# stray pixels. Emitting a 2 mm shell is worse than emitting none.
MIN_THICKNESS = 0.004


class HairSegmenter:
    """Wraps MediaPipe's selfie multiclass segmenter. NOT thread-safe.

    Same constraint as FaceDetector, for the same reason: MediaPipe graphs hold
    state across calls, so two threads sharing one instance corrupt each other.
    Callers that serve concurrent requests need one per thread.
    """

    def __init__(self, model_path=MODEL):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        model_path = pathlib.Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe segmenter missing: {model_path}\n"
                f"  curl -sSL -o {model_path} https://storage.googleapis.com/"
                f"mediapipe-models/image_segmenter/selfie_multiclass_256x256/"
                f"float32/latest/selfie_multiclass_256x256.tflite")

        self._mp = mp
        self._seg = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True))

    def categories(self, image):
        """(H,W) uint8 category ids for an RGB uint8 image."""
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(image))
        # numpy_view() gives (H,W,1); squeeze so callers can index with it.
        return np.squeeze(self._seg.segment(img).category_mask.numpy_view())

    def hair(self, image):
        return self.categories(image) == HAIR

    def close(self):
        self._seg.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def scalp_region(flame, embedding, radius=0.045, min_normal_y=-0.35):
    """(V,) bool over FLAME vertices: the cranium that hair sits on.

    Everything that is not face skin, not an eyeball, above the neck joint, and
    not pointing sharply downward. Each clause removes something that inflating
    would visibly wreck:

      face skin   pushing the face out along its normals is a swollen face
      eyeballs    they are inside the head and bound to their own joints
      neck        FLAME invents a neck and shoulder stub; hair is not on it
      normal y    the ears. They stick out sideways and are not face skin, so
                  they survive every other test and then inflate into blobs

    `radius` matches face_region's default so the hairline lands exactly where
    the skin mask ends, with no gap and no overlap.
    """
    with torch.no_grad():
        v, _ = flame(batch_size=1)
        n = vertex_normals(v, flame.faces)[0]

    face_v = face_region(flame, embedding, radius=radius)
    eye_v = torch.zeros(flame.n_verts, dtype=torch.bool, device=v.device)
    eye_v[flame.faces[eye_faces(flame)].reshape(-1)] = True

    neck_y = 0.0
    if "neck" in JOINT_NAMES:
        from .rig import rest_joints
        neck_y = float(rest_joints(flame, None)[JOINT_NAMES.index("neck")][1])

    return (~face_v & ~eye_v & (v[0][:, 1] > neck_y) & (n[:, 1] > min_normal_y))


def scalp_faces(flame, vertex_mask):
    """Triangles entirely inside the scalp, for the glTF material split."""
    return vertex_mask[flame.faces].all(-1)


def _adjacency(faces, n_verts, device):
    """Sparse vertex-vertex adjacency, for Laplacian smoothing."""
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = torch.cat([e, e.flip(1)], 0)
    val = torch.ones(len(e), device=device)
    A = torch.sparse_coo_tensor(e.t(), val, (n_verts, n_verts)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
    return A, deg


def radial_profile(hair_mask, scalp_px, bins=64, percentile=97.0):
    """Hair thickness in PIXELS, per angle around the cranium.
    -> (profile (bins,), centre (2,)).

    The measurement this whole module rests on, so it is worth being precise
    about what it can and cannot see.

    From one photograph we know the hair's OUTLINE and nothing about its depth.
    So we measure the outline: sweep angles around the cranium centre, and in
    each angular bin compare how far the hair reaches to how far the SCALP
    reaches. The difference is how much hair stands off the skull in that
    direction. Tall hair on top and wide hair at the sides both fall out of
    this correctly; a fringe hanging forward does not, because it is hidden
    behind the face and has no outline of its own.

    `scalp_px`, not the whole projected mesh, and this is not a detail. FLAME
    carries a neck and shoulder stub that reaches far below the head: measured
    on one subject, the mesh spanned y = -57..649 px in a 512 px frame, so the
    downward bins compared hair against a shoulder 409 px from the centre and
    concluded there was no hair anywhere. The scalp is also the surface being
    inflated, so it is the only silhouette the comparison means anything
    against.

    `percentile` rather than max because a single stray segmented pixel -- a
    flyaway strand, a dark background pixel misread as hair -- would otherwise
    set the thickness for its whole bin. At 97 the outline still follows real
    curls, which do reach several percent of their bin.
    """
    ys, xs = np.nonzero(hair_mask)
    centre = np.array([scalp_px[:, 0].mean(), scalp_px[:, 1].mean()])
    if len(xs) == 0:
        return np.zeros(bins, np.float32), centre

    def polar(x, y):
        d = np.stack([x - centre[0], y - centre[1]], -1)
        return np.arctan2(d[:, 1], d[:, 0]), np.linalg.norm(d, axis=-1)

    th_h, r_h = polar(xs.astype(np.float64), ys.astype(np.float64))
    th_v, r_v = polar(scalp_px[:, 0], scalp_px[:, 1])

    idx_h = np.clip(((th_h + np.pi) / (2 * np.pi) * bins).astype(int), 0, bins - 1)
    idx_v = np.clip(((th_v + np.pi) / (2 * np.pi) * bins).astype(int), 0, bins - 1)

    out = np.zeros(bins, np.float32)
    for b in range(bins):
        h, v = r_h[idx_h == b], r_v[idx_v == b]
        if len(h) == 0 or len(v) == 0:
            continue
        out[b] = max(0.0, np.percentile(h, percentile) - v.max())

    # Circular smoothing. The bins are independent measurements of one smooth
    # surface, and un-smoothed they give the shell a scalloped edge.
    k = np.array([0.06, 0.24, 0.4, 0.24, 0.06], np.float32)
    return np.convolve(np.r_[out[-2:], out, out[:2]], k, mode="valid"), centre


def project_px(verts, cam, image_size, ndc_scale=1.0):
    """Project to pixel coordinates in a crop. -> ((V,2) float64, px per unit).

    `ndc_scale` retargets the fitted camera to a WIDER crop of the same face.
    crop_square's frame is set by its margin, and NDC is proportional to it, so
    a crop taken at margin M covers the margin-1.6 frame's NDC scaled by 1.6/M.
    That matters because the hair does not fit in the crop the encoder uses:
    measured on one subject at margin 1.6, the skull top projected to y = -57 in
    a 512 px frame -- above the image -- while the hair was clipped at the top
    edge, so the comparison was between two things the frame had cut.
    """
    from .pipeline import project

    ndc = project(verts, cam)[0] * ndc_scale
    px = ((ndc[:, 0] * 0.5 + 0.5) * image_size).detach().cpu().numpy()
    py = ((0.5 - ndc[:, 1] * 0.5) * image_size).detach().cpu().numpy()
    per_unit = max(float(cam[0, 0]) * ndc_scale * image_size / 2.0, 1e-6)
    return np.stack([px, py], -1).astype(np.float64), per_unit


def shell_offset(flame, verts, cam, hair_mask, scalp, image_size, ndc_scale=1.0,
                 bins=64, taper=0.018, smooth=16, edge_smooth=4,
                 max_thickness=MAX_THICKNESS):
    """Per-vertex outward offset in FLAME units. -> (V,) float tensor.

    verts (1,V,3) posed, cam (1,3), hair_mask (H,W) bool over a crop of the same
    face whose margin is encoded in `ndc_scale`, `scalp` (V,) from scalp_region.

    `taper` is the width, in FLAME units, of the fade to zero at the hairline.
    Without it the shell meets the forehead as a cliff and the head looks like
    it is wearing a bowl. 18 mm is about a finger's width of forehead.
    """
    dev = verts.device
    verts_px, per_unit = project_px(verts, cam, image_size, ndc_scale)
    sc = scalp.detach().cpu().numpy()

    prof, centre = radial_profile(hair_mask, verts_px[sc], bins=bins)
    prof_units = np.clip(prof / per_unit, 0.0, max_thickness)

    # Sample the profile with linear interpolation between bins, wrapping at
    # the seam. Nearest-bin lookup puts a step wherever the angle crosses a bin
    # edge, and a step in an offset field is a ridge on the surface.
    th = np.arctan2(verts_px[:, 1] - centre[1], verts_px[:, 0] - centre[0])
    f = (th + np.pi) / (2 * np.pi) * bins
    lo = np.floor(f).astype(int) % bins
    w = (f - np.floor(f)).astype(np.float32)
    vals = prof_units[lo] * (1 - w) + prof_units[(lo + 1) % bins] * w
    t = torch.as_tensor(vals, dtype=verts.dtype, device=dev)

    s = scalp.to(dev)
    sf = s.to(t.dtype)
    t = t * sf

    # Smooth over the mesh, BEFORE the taper and only across scalp neighbours.
    # The profile is assigned per angle, and vertices near the projected centre
    # -- the back of the head -- get a noisy angle because their radius is tiny,
    # so this is what turns the assignment into a smooth dome.
    #
    # Averaging over ALL neighbours instead collapses the shell. Off-scalp
    # vertices hold zero, so every pass drags the boundary toward zero and
    # diffuses that inward: measured, 12 passes took a 4 cm profile down to a
    # 1 mm mean, which rendered as a head indistinguishable from the bald one.
    # Dividing by the count of SCALP neighbours makes the boundary free rather
    # than pinned at zero, which is the condition we actually want -- the taper
    # below is what brings the edge down, deliberately and over a known width.
    A, _ = _adjacency(flame.faces, flame.n_verts, dev)
    den = torch.sparse.mm(A, sf[:, None])[:, 0].clamp(min=1.0)
    for _ in range(smooth):
        nb = torch.sparse.mm(A, (t * sf)[:, None])[:, 0] / den
        t = (0.5 * t + 0.5 * nb) * sf

    # Fade to nothing at the hairline, so the shell meets the forehead as a
    # slope rather than a cliff. Distance to the nearest non-scalp vertex is a
    # good enough stand-in for a geodesic: the boundary is a closed loop around
    # the skull, so the nearest non-scalp vertex is almost always across it.
    if s.any() and (~s).any():
        d = torch.cdist(verts[0][s], verts[0][~s]).min(dim=1).values
        ramp = (d / taper).clamp(0, 1)
        t[s] = t[s] * (ramp * ramp * (3 - 2 * ramp))            # smoothstep

    # A few UNMASKED passes to finish. Now that the taper has brought the edge
    # down deliberately, letting the zero outside pull on it is what we want:
    # it rounds the hairline, which face_region leaves jagged at triangle
    # resolution, and it takes the corner off the shell where it passes the
    # ear. The ear is not scalp, so it does not inflate, and the scalp beside
    # it was standing off as a horn.
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
    for _ in range(edge_smooth):
        t = 0.5 * t + 0.5 * (torch.sparse.mm(A, t[:, None])[:, 0] / deg)
    return t * (t > 1e-5)


def inflate(verts, faces, offset):
    """Push vertices out along their normals. verts (B,V,3), offset (V,)."""
    n = vertex_normals(verts, faces)
    return verts + n * offset[None, :, None].to(verts.dtype)


def recrop_mask(mask, ratio, size):
    """Re-frame a mask from a wide crop into a narrower, concentric one.

    crop_square centres every crop on the same point, so the margin-M frame is
    the central `ratio` fraction of the margin-M/ratio frame. Cheaper and exactly
    consistent with running the segmenter twice at two margins, which would also
    be free to disagree with itself.
    """
    from PIL import Image

    h = mask.shape[0]
    lo = int(round(h * (1 - ratio) / 2))
    hi = h - lo
    sub = np.ascontiguousarray(mask[lo:hi, lo:hi]).astype(np.uint8) * 255
    return (np.asarray(Image.fromarray(sub).resize((size, size), Image.BILINEAR))
            .astype(np.float32) / 255.0)


def measure(hair_mask, scalp_px, per_unit, bins=64):
    """Peak hair thickness in FLAME units, for the bald/hat decision.

    Peak rather than mean: a receding hairline still has hair, and averaging
    over a profile that is legitimately zero at the front would call it bald.
    """
    prof, _ = radial_profile(hair_mask, scalp_px, bins=bins)
    return float(np.clip(prof / per_unit, 0.0, MAX_THICKNESS).max())
