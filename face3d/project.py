"""Photograph -> UV texture, by running the renderer backwards.

The PCA albedo in `albedo.py` reconstructs a face from 50 numbers. Fifty numbers
cannot encode a mole, a freckle, an eyebrow or stubble -- not "does not yet",
*cannot*, because the basis has no direction that varies at that spatial scale.
Everything it produces is average skin tinted toward the subject, which is why
the output reads as a mannequin.

This module skips the basis. We already fitted geometry and a camera, so we know
where every point of the mesh landed in the photograph -- and therefore what
colour the photograph says that point is. Sampling those pixels gives real skin,
including eyebrows and beard, which FLAME has no geometry for but which read
convincingly as texture.

THE CENTRAL TRICK is to rasterise the mesh *in UV space* rather than in screen
space. Rasterising into the image and scattering pixels back into UV leaves
holes wherever the mesh is drawn smaller than the texture. Rasterising into UV
instead gives every texel exactly one triangle and one set of barycentric
weights; the same weights then interpolate the *screen* positions of that
triangle's corners, telling us where the texel lands in the photograph. Dense by
construction, no holes, one grid_sample.

Three things make it more than a one-liner, and each has a section below:

  visibility  a texel can face the camera and still be hidden behind the nose
  lighting    photograph shadows would otherwise be glued to the face for ever
  coverage    one photo shows at most half a head; the rest must come from
              somewhere

What this CANNOT fix: geometry error. Where the fitted mesh is wrong the texture
smears, because we sample the photo at the place the wrong geometry points to.
Identity shape is this pipeline's known weak point, so expect softness at the
nose and jawline.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .albedo import CACHE_DIR
from .pipeline import project
from .render import interpolate, rasterize, sh_shading, vertex_normals

# Facing weight ramps from 0 to 1 across this band of n.v. Below FACING_MIN the
# surface is so oblique that one texel covers many pixels and the sample is
# stretched mush; above FACING_MAX it is as head-on as this photo will give.
FACING_MIN, FACING_MAX = 0.15, 0.50

# Shading below this is not trustworthy to divide by -- see delight().
SHADE_FLOOR = 0.35


# --------------------------------------------------------------------------
# Static, topology-only precomputation. None of this depends on the subject,
# so it is built once and cached, exactly like face_texel_mask().
# --------------------------------------------------------------------------

def uv_rasterize(vt, ft, resolution=512, device="cpu"):
    """Rasterise the UV unwrap itself. -> (fid (R,R) long, bary (R,R,3), mask).

    `fid` indexes TRIANGLES, and triangle i is the same triangle whether you
    look it up in `ft` (UV corners) or in `flame.faces` (geometry corners) --
    they are parallel arrays in the same corner order. That is what lets the
    barycentric weights found here interpolate geometric attributes later.

    We reuse the screen rasteriser by handing it the UV coordinates as if they
    were vertex positions: x = 2u-1, y = 2v-1, z = 0. The y mapping is chosen so
    that v=1 lands on image row 0, matching FlameTexture.texture()'s output.
    """
    vt = torch.as_tensor(np.asarray(vt), dtype=torch.float32, device=device)
    ft = torch.as_tensor(np.asarray(ft), dtype=torch.long, device=device)

    flat = torch.stack([vt[:, 0] * 2 - 1,
                        vt[:, 1] * 2 - 1,
                        torch.zeros_like(vt[:, 0])], dim=-1)[None]   # (1,Vt,3)

    # K is the per-triangle search window in pixels. The default caps it at 64,
    # which is right for a head drawn in a 224px frame but wrong here: FLAME's
    # unwrap gives the scalp and neck some very large triangles, and a K below
    # their span rasterises them with holes.
    tri = flat[0][ft][..., :2]
    span = (tri.max(-2).values - tri.min(-2).values).max().item() * resolution / 2
    K = max(4, min(int(span) + 4, resolution))

    fid, bary, mask = rasterize(flat, ft, resolution, resolution, K=K)
    return fid[0], bary[0], mask[0]


def mirror_uv_map(flame, vt, ft, resolution=512, device="cpu", max_dist=0.004,
                  normal_agree=0.3):
    """A UV->UV lookup that reflects the head. -> (grid (R,R,2), valid (R,R)).

    `grid` is in grid_sample convention, so sampling any UV-space map with it
    returns that map reflected left-right ACROSS THE SURFACE -- which is NOT the
    same as flipping the image, because FLAME's unwrap is not symmetric in the
    texture plane. Used to fill the cheek the camera never saw with the cheek it
    did. That is a real assumption, not a free lunch: it invents a symmetry the
    subject may not have, and it will happily duplicate a mole.

    Built geometrically rather than combinatorially, because FLAME's topology is
    NOT mirror-symmetric and pretending otherwise fails silently. Measured on
    flame2023_Open: matching mirrored vertices by nearest neighbour gives a
    median error of 0.00115 against a median edge length of 0.00331, and the
    result is neither a permutation nor an involution -- 775 vertices are never
    matched at all, spread across the whole face rather than confined to the
    eyeballs. The template is a mean over real faces, and real faces are
    asymmetric. So there is no exact vertex pairing to find.

    That is fine for this job. We need "which patch of skin is the mirror of
    this one" to within a fraction of a face, not to within a vertex, and the
    symmetry assumption is far cruder than an edge length anyway. So: take each
    texel's 3D position on the template, mirror it, and find the nearest texel
    on the real surface.

    `normal_agree` guards the one failure that would be visible. A nearest point
    in 3D can sit across a thin gap -- upper lip to lower lip, eyelid to
    eyeball -- which would fill a patch of skin with a patch of something else.
    Requiring the mirrored normal to agree keeps the match on the same sheet.
    """
    from scipy.spatial import cKDTree

    fid, bary, mask = uv_rasterize(vt, ft, resolution, device="cpu")
    faces = flame.faces.detach().cpu()
    v = flame.v_template.detach().cpu().float()[None]              # (1,V,3)
    vn = F.normalize(vertex_normals(v, faces), dim=-1, eps=1e-8)

    # Position and normal of the template surface at every texel.
    P = interpolate(v, faces, fid[None], bary[None])[0].numpy().astype(np.float64)
    N = interpolate(vn, faces, fid[None], bary[None])[0].numpy().astype(np.float64)
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12

    ok0 = mask.numpy()
    ys, xs = np.nonzero(ok0)                       # texels that carry surface
    tree = cKDTree(P[ys, xs])
    src_n = N[ys, xs]

    # Mirror the query. A reflected point carries a reflected normal, so the
    # x component of both flips.
    q = P.reshape(-1, 3).copy()
    qn = N.reshape(-1, 3).copy()
    q[:, 0] *= -1.0
    qn[:, 0] *= -1.0

    K = 8
    d, j = tree.query(q, k=K, workers=-1)
    agree = (src_n[j] * qn[:, None, :]).sum(-1) > normal_agree
    agree &= d < max_dist
    hit = agree.any(1)
    pick = np.argmax(agree, axis=1)                # first (nearest) that agrees
    sel = j[np.arange(len(j)), pick]

    # Texel index -> grid_sample coordinate. align_corners=False puts texel
    # centre (i + 0.5) / R at 2*(i + 0.5)/R - 1.
    gx = (xs[sel] + 0.5) / resolution * 2 - 1
    gy = (ys[sel] + 0.5) / resolution * 2 - 1
    grid = np.stack([gx, gy], -1).reshape(resolution, resolution, 2)
    valid = (hit.reshape(resolution, resolution) & ok0)
    grid = np.where(valid[..., None], grid, -2.0)  # -2 samples as zero-pad

    return (torch.as_tensor(grid, dtype=torch.float32, device=device),
            torch.as_tensor(valid, device=device))


def load_static(flame, vt, ft, resolution=512, device="cpu"):
    """uv_rasterize + mirror_uv_map, memoised to disk. Both cost several seconds
    to build and depend only on the topology, so a per-request rebuild would
    dominate a 40 ms reconstruction."""
    cache = CACHE_DIR / ("uv_project_%d.npz" % resolution)
    if cache.exists():
        with np.load(cache) as d:
            return (torch.as_tensor(d["fid"], device=device).long(),
                    torch.as_tensor(d["bary"], device=device),
                    torch.as_tensor(d["mask"], device=device).bool(),
                    torch.as_tensor(d["grid"], device=device),
                    torch.as_tensor(d["valid"], device=device).bool())

    fid, bary, mask = uv_rasterize(vt, ft, resolution, device="cpu")
    grid, valid = mirror_uv_map(flame, vt, ft, resolution, device="cpu")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, fid=fid.numpy(), bary=bary.numpy(),
                        mask=mask.numpy(), grid=grid.numpy(), valid=valid.numpy())
    return (fid.to(device), bary.to(device), mask.to(device),
            grid.to(device), valid.to(device))


# --------------------------------------------------------------------------
# Per-subject projection
# --------------------------------------------------------------------------

def _smoothstep(x, lo, hi):
    t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def delight(rgb, normals, light, weight=None):
    """Undo the photograph's lighting VARIATION. -> (albedo, trust).

    The renderer models a pixel as albedo * shading, so albedo = pixel /
    shading. Skipping this entirely bakes the shadows in: a face lit from the
    left keeps a dark right cheek for ever, and rotating it in the viewer looks
    painted.

    But dividing by the raw shading is wrong in the other direction, and this
    was visible the first time it ran. Albedo and lighting are famously
    inseparable from one photograph, so the encoder is free to explain a warm
    skin tone either as warm skin or as warm light, and it does some of both.
    Dividing that out removes real skin colour along with the lamp, which came
    out as pastel, blotchy faces with the subject's actual tone missing.

    So we normalise the shading to unit mean first, and divide by that. Lighting
    GRADIENT across the face is removed; the overall level and colour are kept,
    which means the baked texture keeps the average skin tone the photograph
    actually shows. The mean is taken over the texels we trust, weighted, so a
    dark occluded corner cannot drag it.

    Still only partly correct, and worth being honest about which parts. Order-2
    spherical harmonics represent smooth ambient light, so the broad gradient
    across a cheek comes out well. It has no model of a cast shadow (a nose
    edge), and none of a specular highlight, so those stay in the texture as
    dark and light patches that will never move again.

    Where shading is small the division amplifies sensor noise without bound,
    so `trust` reports where the result is worth believing rather than silently
    returning a value that looks like data.

    Values stay in the photograph's own 0-1 encoding, NOT converted to linear.
    That matches how the photometric loss was trained -- it compares shaded
    albedo directly against sRGB pixels -- and consistency with the fit matters
    more here than physical correctness.
    """
    shade = sh_shading(normals, light)                       # (B,R,R,3)
    trust = _smoothstep(shade.mean(-1), SHADE_FLOOR * 0.6, SHADE_FLOOR)

    w = trust if weight is None else trust * weight
    w = w.unsqueeze(-1)
    mean = (shade * w).sum((1, 2), keepdim=True) / w.sum((1, 2), keepdim=True).clamp(min=1e-6)
    rel = shade / mean.clamp(min=1e-6)                       # unit-mean shading

    return (rgb / rel.clamp(min=SHADE_FLOOR)).clamp(0.0, 1.0), trust


def project_photo(flame, image, params, verts, static, face_mask=None,
                  screen=512):
    """Sample a photograph into UV space. -> (albedo (R,R,3), weight (R,R)).

    image   (H,W,3) float 0-1, the SAME crop the encoder saw, at any resolution
    params  FlameParams for this subject (needs .cam and .light)
    verts   (1,V,3) posed vertices, as fed to the renderer
    static  the tuple from load_static()
    weight  per-texel confidence in 0-1; 0 means "we learned nothing here"
    """
    fid, bary, uv_mask, _, _ = static
    dev = verts.device
    faces = flame.faces

    # --- where does each texel land in the photograph? --------------------
    # Same barycentric weights, different attribute. This is the whole idea.
    ndc = project(verts, params.cam)                             # (1,V,3)
    tex_ndc = interpolate(ndc, faces, fid[None], bary[None])[0]  # (R,R,3)

    # --- visibility -------------------------------------------------------
    # Facing the camera is necessary but NOT sufficient: at profile the far
    # side of the nose faces the camera and is still hidden by the near side.
    # So render a depth buffer and keep only texels that are the nearest
    # surface along their own ray. Comparing depth rather than triangle id
    # avoids speckle where a neighbouring triangle wins a pixel by rounding.
    s_fid, s_bary, s_mask = rasterize(ndc, faces, screen, screen)
    s_depth = interpolate(ndc[..., 2:], faces, s_fid, s_bary)[0, ..., 0]

    px = ((tex_ndc[..., 0] * 0.5 + 0.5) * screen).long()
    py = ((0.5 - tex_ndc[..., 1] * 0.5) * screen).long()
    inside = (px >= 0) & (px < screen) & (py >= 0) & (py < screen)
    pxc, pyc = px.clamp(0, screen - 1), py.clamp(0, screen - 1)

    zrange = (ndc[..., 2].max() - ndc[..., 2].min()).clamp(min=1e-6)
    near = tex_ndc[..., 2] <= s_depth[pyc, pxc] + 0.02 * zrange
    visible = inside & near & s_mask[0][pyc, pxc] & uv_mask

    # --- sample -----------------------------------------------------------
    # grid_sample wants y DOWN; our NDC has y up, hence the negation.
    grid = torch.stack([tex_ndc[..., 0], -tex_ndc[..., 1]], -1)[None]
    img = image.permute(2, 0, 1)[None].to(dev)                   # (1,3,H,W)
    rgb = F.grid_sample(img, grid, mode="bilinear",
                        align_corners=False, padding_mode="border")
    rgb = rgb[0].permute(1, 2, 0)                                # (R,R,3)

    # --- confidence -------------------------------------------------------
    n = interpolate(vertex_normals(verts, faces), faces, fid[None], bary[None])
    n = F.normalize(n, dim=-1, eps=1e-8)                         # (1,R,R,3)

    # `project` puts the camera on +z looking back down it, so the FLAME-space
    # z component of the normal IS n.v -- no separate view vector needed.
    facing = _smoothstep(n[0, ..., 2], FACING_MIN, FACING_MAX)

    w = visible.to(rgb.dtype) * facing
    if face_mask is not None:
        # Outside the fitted skin region the "photograph" is hair, background
        # or FLAME's invented neck stub. Projecting those produces a head
        # wearing a smear of whatever was behind it. The mask arrives already
        # softened -- a hard edge here reappears as a polygon outline drawn
        # across the forehead, which is exactly what it looked like.
        w = w * torch.as_tensor(face_mask, dtype=w.dtype, device=dev)

    # --- de-light ---------------------------------------------------------
    # After the weights, because the shading average that sets the exposure
    # must be taken over the texels we actually believe.
    albedo, trust = delight(rgb[None], n, params.light, weight=w[None])
    return albedo[0], w * trust[0]


def composite(base, albedo, weight, static, mirror=True, feather=3.0,
              keep=None):
    """Lay the projected texture over the PCA one. -> (R,R,3) float 0-1.

    base    (R,R,3) the PCA texture, same resolution; covers everything
    keep    optional (R,R) in 0-1 that FORCES the base through, used to protect
            the procedurally generated eyes from being painted over

    Order matters. The photograph is better than the basis wherever we actually
    saw the surface, the mirrored photograph is better than the basis but worse
    than the direct sample, and the basis is the only thing that covers the back
    of the head. So: direct where we have it, mirrored where we do not, basis
    underneath both.
    """
    from PIL import Image, ImageFilter

    _, _, _, grid, valid = static
    w = weight.clamp(0, 1)
    out = albedo * w[..., None]
    tot = w.clone()

    if mirror:
        def pull(x):
            x = x.permute(2, 0, 1)[None] if x.dim() == 3 else x[None, None]
            return F.grid_sample(x, grid[None], mode="bilinear",
                                 align_corners=False, padding_mode="zeros")
        m_rgb = pull(albedo)[0].permute(1, 2, 0)
        m_w = pull(w)[0, 0] * valid.to(w.dtype)
        # Only where the direct sample is weak, and never at full strength --
        # a mirrored cheek is a guess, and it should lose to a real one.
        m_w = m_w * (1.0 - w) * 0.9
        out = out + m_rgb * m_w[..., None]
        tot = tot + m_w

    out = out / tot.clamp(min=1e-6)[..., None]

    # Feather the alpha so the edge of what the camera saw is a gradient, not a
    # visible outline of the photograph's silhouette baked into the skin.
    a = tot.clamp(0, 1).detach().cpu().numpy()
    if feather > 0:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(feather)))
        a = a.astype(np.float32) / 255.0
    if keep is not None:
        a = a * (1.0 - np.asarray(keep, np.float32))
    a = torch.as_tensor(a, device=out.device, dtype=out.dtype)[..., None]

    return (out * a + base * (1.0 - a)).clamp(0, 1)
