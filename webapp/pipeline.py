"""Photograph bytes -> GLB bytes. No HTTP anywhere in this file.

Keeping the domain logic free of web concerns means you can test the hard part
from a plain Python prompt:

    from webapp.pipeline import Reconstructor
    r = Reconstructor()
    glb = r.reconstruct(open("photo.jpg", "rb").read())
    open("out/test.glb", "wb").write(glb)
    # then: node scripts/validate_glb.js out/test.glb

When the web layer later misbehaves you will already know the model half is fine.

Reference: scripts/export_head.py does all of this in order, as a CLI.
"""

import io
import pathlib
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]

from face3d import assets
from face3d.albedo import CACHE_DIR, FlameTexture, face_texel_mask, harmonise
from face3d.detect import FaceDetector, crop_square
from face3d.encoder import ResNetEncoder
from face3d.facemask import eye_faces
from face3d.flame_torch import FlameTorch
from face3d.gltf import build_gltf, glb_bytes, vertex_normals
from face3d.landmarks import LandmarkEmbedding
from face3d.hair import (FACE_SKIN, HAIR, MIN_THICKNESS, HairSegmenter,
                         inflate, measure, project_px, scalp_faces,
                         scalp_region, shell_offset)
from face3d.project import (FACING_MAX, FACING_MIN, composite,
                            frequency_merge, load_static, project_photo)
from face3d.regions import crease_map, region_masks
from face3d.skin import compose as skin_compose
from face3d.skin import freckle_noise, split as skin_split
from face3d.texgen import TextureGenerator
from face3d.rig import JOINT_NAMES, expression_targets, jaw_target, rest_joints

# Must match training. Every corpus and the NoW evaluation crop at 1.6, and a
# mismatch is a distribution shift, not a cosmetic difference. See MODEL_CARD.md.
CROP_MARGIN = 1.6

# How many expression blendshapes to export. FLAME has 100; most real-time
# engines choke well before that, and the tail components move vertices by
# fractions of a millimetre. 20 + jaw_open is a reasonable rig.
MORPH_TARGETS = 20

# Cut between "detail, take it from the photograph" and "tone, take it from
# the basis", as a fraction of the texture. See project.frequency_merge.
MERGE_SIGMA = 0.045
MERGE_STRENGTH = 0.85

# Baked texture resolution when projecting the photograph (face3d/project.py).
# 256 is right for the PCA basis, which has no detail above that scale anyway;
# a projected texture carries real pores and eyebrows, so it earns the pixels.
PROJECT_RES = 512

# The crop the photograph is SAMPLED from. crop_square's NDC mapping does not
# depend on its `size`, so this is the same field of view the encoder saw, just
# not thrown away first: 224 is what the model needs to look at, not what the
# texture needs to read from.
PROJECT_CROP = 1024

# Resolution of the depth buffer used for the occlusion test. Independent of
# the texture: it only answers "is this texel hidden behind something", and the
# head spans ~200 px however large the texture is. Measured at 512 it costs
# 0.200 s, at 256 it costs 0.066 s, and the results are indistinguishable.
PROJECT_SCREEN = 256

# Hair does not fit in the crop the encoder uses. At margin 1.6 the skull top
# projects ABOVE the frame on some subjects while the hair is clipped by the
# top edge, so the silhouette comparison compares two things the crop has cut.
# crop_square's NDC is proportional to its margin, so the fitted camera
# retargets to the wider frame by scaling NDC by 1.6/3.0.
HAIR_MARGIN = 3.0
HAIR_SEG = 512
HAIR_RATIO = CROP_MARGIN / HAIR_MARGIN

# Resolution the occluder mask is segmented at. The model works at 256
# internally and the mask is sampled through grid_sample, so it does not have
# to match the photograph.
SEG_SIZE = 512

# Resolution the procedural skin layer composes at. Must match what the
# generator was trained against -- see the note where it is loaded.
SKIN_RES = 256


class NoFaceFound(Exception):
    """No face in the image. A USER error, not a server fault -- the caller
    should turn this into a 422, never a 500."""


class Reconstructor:
    """Built ONCE per process, reused for every request.

    Loading takes ~4 seconds; running takes ~40 milliseconds. Anything that
    constructs this per request is 100x slower than it needs to be, and that is
    the most common mistake in ML web services.

    deca_open is the default: the only licence-clean checkpoint (FLAME 2023
    Open + DigiFace + FFHQ, no CelebA, no FLAME 2020), the only one whose
    exported meshes may be redistributed, and the only one with predicted eye
    colour. It gives up nothing measurable to be all three.

    THE CHECKPOINT AND THE FLAME BASIS MUST MATCH, and nothing enforces that
    automatically. face3d/assets.py resolves FLAME by searching CANDIDATES,
    which lists FLAME2020 FIRST, so leaving it implicit would silently drive
    deca_open's weights through the 2020 basis. The two bases share topology and
    dimensions, so nothing would raise -- but their axes are rotated, retaining
    only 81% of each other's energy, and the output would be quiet nonsense.
    Hence flame_model is explicit here rather than inherited from list order.
    """

    # Matched pair. Change both or neither.
    CHECKPOINT = ROOT / "runs" / "deca_open" / "encoder.pt"
    FLAME_MODEL = "FLAME2023Open/flame2023_Open.pkl"
    TEXGEN = ROOT / "runs" / "texgen_gen6" / "model.pt"

    def __init__(self, checkpoint=None, device="cpu", flame_model=None,
                 project=True, hair=False, texgen=True):
        self.device = device
        self.project = project
        # The learned texture generator (face3d/texgen.py). When on it
        # REPLACES projection rather than layering on it: the point is that
        # its output is on-manifold by construction, and compositing a raw
        # projection over the top would hand back the artefacts it exists to
        # exclude. It is also much faster -- no projection, no segmentation.
        self.texgen = None
        self.texgen_path = texgen
        self.skin_masks = None
        self.freckles = None
        self.crease = None
        # Parameters predicted for the most recent reconstruction, so a caller
        # can read what the photograph implied before overriding any of it.
        self.last_params = None
        self.skin_procedural = False
        # OFF by default. The shell is a cap fitted to one view's silhouette,
        # and on real photographs it reads worse than leaving the head bald --
        # it cannot follow a hairstyle, so it lands in the valley between "no
        # hair" and "that person's hair". Kept behind the flag rather than
        # deleted because the measurement works; the representation is what is
        # wrong, and that needs real hair geometry, not a better fit.
        #
        # Also rides on projection: the shell is only worth anything if the
        # photograph's own hair can be painted onto it.
        self.hair = hair and project
        checkpoint = pathlib.Path(checkpoint or self.CHECKPOINT)
        flame_model = flame_model or self.FLAME_MODEL

        # FLAME: the frozen statistical face model. Given (shape, expression,
        # pose) coefficients it produces 5023 vertices. Nothing here is trained
        # -- it is a fixed basis the encoder learns to drive.
        flame_path = assets.model_path(flame_model)
        if flame_path is None:
            raise FileNotFoundError(
                f"FLAME basis {flame_model!r} not found. It must match the "
                f"checkpoint: {checkpoint.parent.name} was trained on it, and a "
                f"mismatch produces plausible-looking nonsense rather than an "
                f"error.")
        print(f"    FLAME: {flame_path.name}   checkpoint: {checkpoint.parent.name}",
              flush=True)
        self.flame = FlameTorch(flame_path).to(device)

        # The encoder: ResNet-50 trunk + one linear head emitting 186 numbers
        # (100 shape, 50 expression, 6 pose, 3 camera, 27 light, 50 albedo).
        # pretrained=False because we are about to overwrite every weight with
        # the checkpoint; downloading ImageNet weights first would just be slow.
        self.enc = ResNetEncoder(n_shape=100, n_expr=50,
                                 pretrained=False).to(device)
        self.enc.load_state_dict(
            torch.load(checkpoint, map_location=device)["model"])

        # .eval() switches BatchNorm to its running statistics and disables
        # dropout. Forgetting it makes results depend on batch composition --
        # a single-image request would normalise against itself and produce
        # garbage that still looks plausible.
        self.enc.eval()

        # The albedo basis: 50 PCA components over face textures, which turn
        # the predicted albedo coefficients into a 256x256 UV colour map.
        # Optional -- without it the GLB is geometry with no material.
        cache = CACHE_DIR / "flame_texture_256_50.npz"
        self.texture = FlameTexture(cache, device=device) if cache.exists() else None
        if self.texture is not None:
            self.texture.attach_eyes(self.flame)
        self.uv = self._uv_layout(cache)
        self.uv_split = self._uv_split(cache)

        # Resolution the GLB's texture is baked at. Projection reads real detail
        # out of the photograph, so it gets more pixels than the PCA basis can
        # justify on its own.
        self.tex_res = PROJECT_RES if self.project else 256

        # Which UV texels the photometric loss actually optimised. Depends only
        # on FLAME and the landmark embedding, so it is built once here rather
        # than per request. Used twice: to stop projection painting hair and
        # background onto the head, and by harmonise() -- see there for why.
        self.face_mask = None
        self.static = None
        self.eye_keep = None
        self.proj_mask = None
        self.scalp = None
        self.hair_faces = None
        self.proj_mask_hair = None
        self.hair_texel = None
        self.facing_lo = None
        self.facing_hi = None
        if cache.exists():
            emb = LandmarkEmbedding(
                ROOT / "mediapipe_landmark_embedding" /
                "mediapipe_landmark_embedding.npz", device=device)
            with np.load(cache) as d:
                vt = d["vt"].astype(np.float32)
                ft = d["ft"].astype(np.int64)
            self.face_mask = face_texel_mask(self.flame, emb, vt, ft,
                                             resolution=self.tex_res)

            if self.project:
                # UV-space rasterisation and the mirror correspondence. Topology
                # only, so it is cached to disk and shared by every request --
                # ~1.4 s to build against a ~40 ms reconstruction.
                self.static = load_static(self.flame, vt, ft,
                                          resolution=self.tex_res, device=device)
                self.proj_mask = self._soften(self.face_mask)
                # Protect the generated eyes from being painted over. The
                # eyeball is a sphere posed by a predicted gaze, so projecting
                # the photograph onto it would smear an iris across the sclera
                # wherever that gaze is even slightly off -- and the procedural
                # iris already tracks the photo at r = +0.822.
                a = self.texture.eye_alpha                      # (1,1,256,256)
                a = F.interpolate(a, size=(self.tex_res, self.tex_res),
                                  mode="bilinear", align_corners=False)
                self.eye_keep = a[0, 0].cpu().numpy()

            if self.hair:
                # The cranium, and the triangles covering it. Topology only.
                self.scalp = scalp_region(self.flame, emb)
                self.scalp_np = self.scalp.cpu().numpy()
                self.hair_faces = scalp_faces(self.flame,
                                              self.scalp).cpu().numpy()
                # Which UV texels the shell occupies. The rasterised fid map
                # from load_static() already answers this exactly, so there is
                # no need for a second polygon rasterisation.
                fid = self.static[0].cpu().numpy()
                uvm = self.static[2].cpu().numpy()
                tex_scalp = (self.hair_faces[np.clip(fid, 0, None)] & uvm)
                # When hair is present the photograph is allowed onto the scalp
                # as well as the face. Without this the shell is geometry with
                # the albedo basis's bald scalp painted on it, which is worse
                # than no shell at all.
                self.proj_mask_hair = self._soften(
                    np.maximum(self.face_mask, tex_scalp.astype(np.float32)))
                # harmonise() must not repaint the shell with skin, but hair
                # must not tint the fill colour either, so it is `keep`, not
                # part of the mask.
                self.hair_texel = self._soften(tex_scalp.astype(np.float32),
                                               erode=0.0, blur=0.02)
                # The crown grazes the camera, so the default facing band
                # rejects it outright. Give the scalp its own, much lower.
                sc = tex_scalp.astype(np.float32)
                self.facing_lo = FACING_MIN + sc * (0.02 - FACING_MIN)
                self.facing_hi = FACING_MAX + sc * (0.25 - FACING_MAX)

        if self.texgen_path:
            path = pathlib.Path(self.TEXGEN if texgen is True else texgen)
            if not path.exists():
                # Fall back rather than raise. Unlike a FLAME/checkpoint
                # mismatch, which produces plausible nonsense and must fail
                # loudly, projection is a working alternative -- so say what
                # happened and carry on. runs/ is gitignored, so a fresh clone
                # legitimately arrives without this file.
                print(f"    NOTE: no texture generator at {path}; falling back "
                      f"to projection. Train it with scripts/train_texgen.py.",
                      flush=True)
                self.texgen_path = None
            else:
                ck = torch.load(path, map_location=device)
                self.texgen = TextureGenerator(ck["latent"], 256, ck["width"],
                                               pretrained=False).to(device)
                # Checkpoints trained before the procedural layer have no
                # params head. Load what is there and compose only when the
                # weights to drive it exist -- a strict load would raise a bare
                # "Missing key(s)" that says nothing about what to do.
                missing, unexpected = self.texgen.load_state_dict(ck["model"],
                                                                  strict=False)
                if unexpected:
                    raise RuntimeError(
                        f"texture generator {path} has unexpected weights "
                        f"{unexpected[:4]}; it does not match this code")
                self.skin_procedural = not any(m.startswith("params.")
                                               for m in missing)
                self.texgen.eval()
                print(f"    texture generator: {path.parent.name} "
                      f"(val {ck['val']:.5f})", flush=True)

                # Fixed inputs for the procedural layer, at the resolution the
                # generator was TRAINED at, not the bake resolution. The
                # freckle field is a specific noise image and the model learned
                # an amount against it; regenerating it larger would be a
                # different field, and the amount would no longer mean the same
                # thing. Compose at 256, upsample the result.
                st = load_static(self.flame, vt, ft, resolution=SKIN_RES,
                                 device=device)
                self.skin_masks = {k: v.to(device).float() for k, v in
                                   region_masks(self.flame, emb, st,
                                                device=device).items()}
                self.freckles = torch.as_tensor(freckle_noise(SKIN_RES),
                                                device=device)
                self.crease = torch.as_tensor(
                    crease_map(ROOT / "data" / "texgen_cache", st, SKIN_RES),
                    device=device)

        # MediaPipe graphs are stateful and NOT safe to share across threads.
        # FastAPI runs sync handlers in a worker threadpool, so two concurrent
        # requests would otherwise call detect() on one object at the same time.
        # threading.local() gives each thread its own; ingest_ffhq.py needed the
        # same fix for the same reason.
        # Eyeball triangles, so they can be exported as their own primitive
        # with a glossy material. Depends only on FLAME, so compute it once.
        self.eye_mask = eye_faces(self.flame).cpu().numpy()

        self._local = threading.local()

    def _uv_split(self, cache):
        """The UV unwrap without collapsing its seams. -> (uv2v, uv) or None.

        glTF allows exactly one UV per vertex, and FLAME's unwrap needs 5118 for
        5023 vertices: a seam is a cut where one vertex must appear at two
        places in the texture. `_uv_layout` resolves that by letting the last
        writer win, which is wrong for the handful of corners on a seam.

        "A handful" is 284 triangle corners, 0.95% of the mesh, and it is not
        invisible. They sit on the inner-mouth seam, so the wrong UV samples
        somewhere unrelated in the texture and the GLB renders a bright patch
        inside the mouth -- visible from below, which is exactly where nobody
        looks until they do. It never showed up in any diagnostic render here
        because those sample through the correct per-corner UVs; only the
        export was affected.

        The fix is the standard one: duplicate the seam vertices so each side
        gets its own, which is what the unwrap always intended. Costs 95 extra
        vertices, 1.9%.

        NORMALS MUST STILL COME FROM THE UNSPLIT MESH. Recomputing them after
        the split would treat the seam as a boundary and shade it as a crease,
        trading an invisible texture bug for a visible lighting one.
        """
        if not cache.exists():
            return None
        with np.load(cache) as d:
            vt = d["vt"].astype(np.float32)          # (5118, 2)
            ft = d["ft"].astype(np.int64)            # (9976, 3)

        faces = self.flame.faces.cpu().numpy()
        uv2v = np.zeros(len(vt), np.int64)
        for corner in range(3):
            uv2v[ft[:, corner]] = faces[:, corner]

        uv = vt.copy()
        uv[:, 1] = 1.0 - uv[:, 1]                    # OBJ bottom-up -> glTF top-down
        return uv2v, uv, ft

    def _soften(self, mask, erode=0.02, blur=0.035):
        """Turn the hard skin mask into a wide ramp, for use as a blend alpha.

        face_texel_mask() rasterises whole triangles, so its boundary is a
        polygon with visible straight edges. Used directly as the projection
        weight that boundary is drawn onto the face: the first render showed a
        crisp polygonal outline across the forehead where the photograph
        stopped and the basis took over.

        Eroding first, then blurring, keeps the soft ramp INSIDE the region the
        loss actually optimised, rather than smearing the projection outwards
        into the hair. Both radii are fractions of the texture, so this behaves
        the same at any resolution.
        """
        from PIL import Image, ImageFilter

        R = mask.shape[0]
        img = Image.fromarray((mask * 255).astype(np.uint8))
        for _ in range(int(erode * R / 4)):
            img = img.filter(ImageFilter.MinFilter(5))       # ~2 px a pass
        img = img.filter(ImageFilter.GaussianBlur(blur * R))
        return np.asarray(img).astype(np.float32) / 255.0

    def _segmenter(self):
        """One HairSegmenter per thread. Same rule as _detector: MediaPipe
        graphs are stateful, so two requests sharing one corrupt each other."""
        if not hasattr(self._local, "seg"):
            self._local.seg = HairSegmenter()
        return self._local.seg

    def _detector(self):
        """One FaceDetector per thread, created lazily on first use."""
        if not hasattr(self._local, "det"):
            # blendshapes=False: MediaPipe can also regress 52 ARKit-style
            # expression scores, which we do not use and which cost time.
            self._local.det = FaceDetector(blendshapes=False)
        return self._local.det

    def _uv_layout(self, cache):
        """One UV coordinate per vertex, in glTF's orientation. (V,2) or None.

        The texture cache stores FLAME's UV unwrap as `vt` (the UV coordinates)
        and `ft` (which UV belongs to each triangle corner). That is OBJ's
        convention, and it does not line up with glTF's in two ways.

        1. vt has 5118 entries but the mesh has 5023 vertices. UV seams -- the
           cuts where the unwrap is split open -- duplicate a vertex so its two
           sides can land in different parts of the texture. glTF allows exactly
           one UV per vertex, so each vertex takes the UV of the first triangle
           corner that references it. Wrong only for triangles sitting exactly
           on a seam, which is a handful and invisible.

        2. OBJ puts v=0 at the BOTTOM of the image; glTF puts it at the TOP.
           Miss the flip and the head loads with its face upside down -- and
           still passes the validator, because the file is well-formed. This is
           the kind of bug you find by looking, never by testing return codes.
        """
        if not cache.exists():
            return None
        with np.load(cache) as d:
            vt = d["vt"].astype(np.float32)      # (5118, 2) UV coordinates
            ft = d["ft"].astype(np.int64)        # (9976, 3) UV index per corner

        faces = self.flame.faces.cpu().numpy()   # (9976, 3) vertex index per corner
        uv = np.zeros((self.flame.n_verts, 2), np.float32)
        for corner in range(3):
            # Scatter: for every triangle, vertex faces[:,c] gets UV vt[ft[:,c]].
            # Later writes overwrite earlier ones, so effectively "last writer
            # wins" -- which is fine, since only seam vertices disagree.
            uv[faces[:, corner]] = vt[ft[:, corner]]

        uv[:, 1] = 1.0 - uv[:, 1]                # OBJ bottom-up -> glTF top-down
        return uv

    def _bake_texture(self, albedo, eye=None, photo=None, params=None,
                      verts=None, hair=False, photo_mask=None, crop=None,
                      skin_override=None):
        """Albedo coefficients (+ optionally the photograph) -> PNG bytes.

        Two sources, layered. FlameTexture.texture() evaluates
        mean + sum(coeff_i * basis_i), which covers the whole head but carries
        no detail: 50 numbers cannot encode a mole or an eyebrow. When the
        photograph is available we sample it directly into UV space and lay that
        over the top, keeping the basis underneath for everything the camera
        never saw. See face3d/project.py.

        Everything stays in memory and is encoded as PNG for the GLB's binary
        chunk, so the asset is one self-contained file with no external refs.
        """
        if self.texture is None or self.uv is None:
            return None
        with torch.no_grad():
            # albedo is (50,); texture() wants a batch, hence [None] -> (1,50).
            tex = self.texture.texture(albedo[None].to(self.device),
                                       eye=eye[None].to(self.device))

            if self.texgen is not None and crop is not None:
                # Learned residual, then the procedural layer on top of it --
                # the same order as training, at the same resolution as
                # training. Composing at the bake resolution instead would put
                # the freckle field on a different noise image from the one the
                # predicted amount was learned against.
                res, raw = self.texgen(crop.to(self.device))
                self.last_params = skin_split(raw)
                p = dict(self.last_params)
                if skin_override:
                    # The character-creator hook: any named parameter can be
                    # replaced without touching the rest of the face.
                    for k, v in skin_override.items():
                        if k not in p:
                            raise KeyError(
                                f"unknown skin parameter {k!r}; have "
                                f"{sorted(p)}")
                        p[k] = torch.as_tensor(
                            v, dtype=res.dtype, device=self.device
                        ).reshape(1, -1)
                # Unclamped into compose(), which clamps once at the end.
                # The residual deliberately overshoots where the procedural
                # layer will darken; clipping between them leaves the overshoot
                # and removes the correction, which showed as white arcs above
                # the brows.
                tex = tex + res
                if self.skin_procedural:
                    tex = skin_compose(tex, p, self.skin_masks,
                                       self.freckles, self.crease)
                tex = tex.clamp(0, 1)

            if tex.shape[-1] != self.tex_res:
                tex = F.interpolate(tex, size=(self.tex_res, self.tex_res),
                                    mode="bilinear", align_corners=False)
            # (1,3,R,R) -> (R,R,3), the layout everything downstream expects.
            base = tex[0].permute(1, 2, 0).clamp(0, 1)

            w = None
            if photo is not None and self.static is not None and self.texgen is None:
                mask = self.proj_mask_hair if hair else self.proj_mask
                lo = self.facing_lo if hair else FACING_MIN
                hi = self.facing_hi if hair else FACING_MAX
                alb, w = project_photo(self.flame, photo, params, verts,
                                       self.static, face_mask=mask,
                                       screen=PROJECT_SCREEN,
                                       facing_min=lo, facing_max=hi,
                                       photo_mask=photo_mask)
                merged = frequency_merge(alb.cpu().numpy(),
                                         base.cpu().numpy(),
                                         sigma=MERGE_SIGMA * self.tex_res,
                                         strength=MERGE_STRENGTH)
                alb = torch.as_tensor(merged, dtype=alb.dtype, device=alb.device)
                base = composite(base, alb, w, self.static,
                                 feather=0.012 * self.tex_res,
                                 keep=self.eye_keep)

        arr = base.cpu().numpy()

        # The back of the head is hair the camera never saw. Left alone the
        # composite falls through to the albedo basis there, which is a BALD
        # scalp -- so the shell came out with hair on top and skin behind it,
        # meeting at a seam. Fill the unseen scalp with the mean of the hair we
        # did see. Same argument as harmonise(): the fill is a colour actually
        # measured on this person, not one invented.
        if hair and w is not None and self.hair_texel is not None:
            sc = self.hair_texel > 0.5
            got = sc & (w.cpu().numpy() > 0.25)
            if got.any():
                fill = arr[got].mean(0)
                # max(), not the blur alone. Blurring `got` on its own pulls
                # the alpha below 1 INSIDE the measured region too, so the real
                # hair gets averaged toward its own mean and comes out flat and
                # pale -- which is exactly what the first attempt rendered.
                # Measured texels keep full weight; the blur only ramps outward.
                a = np.maximum(got.astype(np.float32),
                               self._soften(got.astype(np.float32),
                                            erode=0.0, blur=0.03))
                a = np.where(sc, a, 1.0)[..., None]
                arr = arr * a + fill[None, None, :] * (1.0 - a)

        # Replace the un-fitted neck and scalp with the fitted face's own tone.
        # Without this the basis mean shows through at 27% higher saturation
        # than the subject, and a correct face reads as washed out beside it.
        # Scale the falloff with resolution so it stays the same width of face.
        # Not when the generator is driving. harmonise() replaces everything
        # outside the face mask with the fitted face's mean tone, which is a
        # hand-made approximation of exactly what the generator now learns --
        # it extends the correction over the neck and jaw itself. Applying both
        # would flatten that extension back to a constant and reinstate the
        # boundary the extension exists to remove.
        if self.face_mask is not None and self.texgen is None:
            arr = harmonise(arr, self.face_mask,
                            blur=10.0 * self.tex_res / 256,
                            keep=self.hair_texel if hair else None)

        arr = (arr * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def reconstruct(self, image_bytes: bytes) -> bytes:
        """Photograph -> GLB bytes. Nothing touches the filesystem.

        Uploaded faces are biometric data under GDPR and BIPA. The easiest way
        to honour "do not retain it" is to have no code path that could.
        """
        return glb_bytes(*self.build(image_bytes))

    def build(self, image_bytes: bytes, targets: int = MORPH_TARGETS,
              skin_override=None):
        """Photograph -> (gltf dict, binary blob), the pieces before packing.

        Split out from reconstruct() so scripts/export_head.py can write .gltf
        alongside .glb without owning a second copy of this pipeline. It had
        one, and the copy had drifted: it resolved FLAME implicitly, which
        picks FLAME 2020 whatever the checkpoint was trained on.
        """
        # --- 1. decode -------------------------------------------------------
        # convert("RGB") normalises away greyscale, palettes and RGBA. Without
        # it a PNG with an alpha channel arrives as (H,W,4) and every later
        # shape assumption breaks.
        img = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

        # --- 2. find the face ------------------------------------------------
        # res["norm"] is (478,2) landmarks in normalised [0,1] image coords.
        res = self._detector().detect(img)
        if res is None:
            raise NoFaceFound("no face detected in the image")

        # --- 3. crop ---------------------------------------------------------
        # A square crop centred on the landmark bounding box, expanded by
        # CROP_MARGIN and resized to 224x224 -- exactly what the encoder saw in
        # training. Passing margin explicitly rather than relying on the default
        # so a reader can see it matches without checking the signature.
        crop, _ = crop_square(img, res["norm"], size=224, margin=CROP_MARGIN)

        # --- 4. to a tensor --------------------------------------------------
        # ascontiguousarray because crop_square can hand back a view with
        # negative strides, which torch.from_numpy refuses.
        # (224,224,3) uint8 -> permute to (3,224,224) -> [None] adds the batch
        # dimension -> (1,3,224,224) float in [0,1]. The encoder applies its own
        # ImageNet mean/std normalisation internally, so do not do it here.
        x = torch.from_numpy(np.ascontiguousarray(crop))
        x = x.to(self.device).permute(2, 0, 1)[None].float() / 255.0

        # --- 5. predict ------------------------------------------------------
        # no_grad() stops autograd building a graph we will never backward
        # through: less memory, faster.
        with torch.no_grad():
            # calibrate=True multiplies shape by SHAPE_CALIBRATION (0.40).
            # NOT optional: the raw prediction scores 1.5485 mm on NoW, WORSE
            # than emitting the FLAME mean face (1.3554). The shrinkage is the
            # only reason the model beats a constant mesh.
            p = self.enc.predict(x, calibrate=True)

            # The posed mesh the encoder actually fitted. Only needed to project
            # the photograph, which has to sample the image at the place THIS
            # geometry says each texel landed -- the neutral export mesh below
            # is a different pose and would sample the wrong pixels.
            verts = None
            if self.project and self.static is not None:
                pp = p.pad_to(self.flame.n_shape, self.flame.n_expr)
                verts, _ = self.flame(pp.shape, pp.expr, pp.pose)

        # --- 5b. hair --------------------------------------------------------
        # Segment the photograph, measure how far the hair stands off the skull,
        # and push the scalp out to meet it. See face3d/hair.py -- this is a cap
        # fitted to one view's silhouette, not a hairstyle.
        # --- 5b. what may be sampled ----------------------------------------
        # Gate the projection on the segmenter's face-skin class. This is the
        # single biggest quality fix available, because the dominant artefact
        # is not smearing -- it is OCCLUDERS being painted onto the face.
        # Audited over 14 FFHQ faces, 8 carried one: spectacle frames, a fringe
        # across the forehead, a beaded headdress, hair blown over a cheek.
        # Projection has no notion of "this pixel is not skin", so it transfers
        # them faithfully onto a head that cannot represent them.
        #
        # The class boundary happens to fall exactly where we need it: eyebrows
        # and lips are face-skin, so they survive, while glasses and headdresses
        # are `other` and hair over the forehead is `hair`, so they do not.
        photo_mask = None
        if verts is not None:
            seg_crop, _ = crop_square(img, res["norm"], size=SEG_SIZE,
                                      margin=CROP_MARGIN)
            cats = self._segmenter().categories(seg_crop)
            allow = cats == FACE_SKIN
            if self.hair:
                # The shell needs hair pixels, which are the very thing the
                # face gate rejects. Opt-in, and it does re-admit a fringe.
                allow = allow | (cats == HAIR)
            photo_mask = self._soften(allow.astype(np.float32),
                                      erode=0.0, blur=0.004)

        # --- 5c. hair --------------------------------------------------------
        hair_off = None
        if verts is not None and self.hair:
            wide, _ = crop_square(img, res["norm"], size=HAIR_SEG,
                                  margin=HAIR_MARGIN)
            hair_px = self._segmenter().categories(wide) == HAIR
            vpx, per_unit = project_px(verts, p.cam, HAIR_SEG, HAIR_RATIO)
            peak = measure(hair_px, vpx[self.scalp_np], per_unit)
            # Bald, or a hat, or a handful of stray pixels. A 2 mm shell reads
            # as a swollen skull, which is worse than leaving the head bald.
            if peak >= MIN_THICKNESS:
                with torch.no_grad():
                    hair_off = shell_offset(self.flame, verts, p.cam, hair_px,
                                            self.scalp, HAIR_SEG,
                                            ndc_scale=HAIR_RATIO)
                    # Texture is projected through the INFLATED mesh, so the
                    # photograph's hair pixels land on the shell that now
                    # reaches them.
                    verts = inflate(verts, self.flame.faces, hair_off)

        # From here on we are building the ASSET, not running the model, so
        # everything moves to numpy. [0] drops the batch dimension.
        shape = p.shape[0].cpu().numpy()          # (100,) identity coefficients

        # --- 6. blendshapes --------------------------------------------------
        # expression_targets returns, for THIS person's identity:
        #   deltas  (20, 5023, 3) per-vertex offsets, one per expression
        #   names   the target names, expr_00 .. expr_19
        #   neutral (5023, 3) the rest mesh the deltas are offsets FROM
        # glTF morph targets are offsets, not absolute positions, which is why
        # the neutral mesh and the deltas travel together.
        faces_np = self.flame.faces.cpu().numpy()
        deltas, names, neutral = expression_targets(self.flame, shape,
                                                    n_targets=targets)

        # The jaw is a JOINT in FLAME, not an expression component, so opening
        # the mouth is not reachable through the expression basis alone. We bake
        # a jaw rotation into an extra blendshape so the rig can open its mouth
        # without the consumer needing to pose the skeleton. [None] makes it
        # (1,5023,3) so it concatenates onto the 20 expression deltas.
        deltas = np.concatenate([deltas, jaw_target(self.flame, shape)[None]], 0)
        names = names + ["jaw_open"]

        # --- 7. skeleton -----------------------------------------------------
        # Joint positions depend on identity: a larger head puts the neck joint
        # somewhere different. (5,3) for root/neck/jaw/eye_left/eye_right.
        joints = rest_joints(self.flame, shape).cpu().numpy()

        # The same offsets on the exported rest mesh. Morph targets are DELTAS
        # from it, and expression moves the scalp by almost nothing, so they
        # stay valid unchanged. Normals are recomputed on the neutral pose
        # because that is the surface being displaced here.
        if hair_off is not None:
            off = hair_off.cpu().numpy()[:, None]
            neutral = neutral + vertex_normals(neutral, faces_np) * off

        # --- 8. texture ------------------------------------------------------
        # Re-crop the SAME box at higher resolution. crop_square's NDC mapping
        # is independent of `size`, so this is the identical field of view the
        # encoder saw and the fitted camera still projects into it correctly --
        # we simply stopped throwing the pixels away. 224 is what the model
        # needs to look at; the texture wants everything the photo has.
        photo = None
        if verts is not None:
            hi, _ = crop_square(img, res["norm"], size=PROJECT_CROP,
                                margin=CROP_MARGIN)
            photo = torch.from_numpy(np.ascontiguousarray(hi))
            photo = photo.to(self.device).float() / 255.0

        tex_png = self._bake_texture(p.albedo[0], p.eye[0], photo=photo,
                                     params=p, verts=verts,
                                     hair=hair_off is not None,
                                     photo_mask=photo_mask, crop=x,
                                     skin_override=skin_override)

        # --- 9. assemble ------------------------------------------------------
        # build_gltf returns (json_dict, binary_blob); glb_bytes packs them into
        # the GLB container with the 4-byte chunk padding the spec requires.
        # skin_weights (5023,5) says how much each joint influences each vertex;
        # build_gltf keeps the top 4 because glTF's JOINTS_0 is a vec4.
        # Emit the seam-split mesh so no triangle corner carries a UV that
        # belongs to the other side of a seam. See _uv_split.
        verts_out, faces_out, uv_out = neutral, faces_np, self.uv
        normals_out = None
        weights_out = self.flame.weights.cpu().numpy()
        deltas_out = deltas
        if self.uv_split is not None:
            uv2v, uv_out, faces_out = self.uv_split
            normals_out = vertex_normals(neutral, faces_np)[uv2v]
            verts_out = neutral[uv2v]
            weights_out = weights_out[uv2v]
            deltas_out = deltas[:, uv2v]

        gltf, blob = build_gltf(
            verts=verts_out,
            faces=faces_out,
            normals=normals_out,
            joints=joints,
            parents=self.flame.parents,
            skin_weights=weights_out,
            joint_names=JOINT_NAMES,
            morph_targets=deltas_out,
            morph_names=names,
            uv=uv_out,
            name="face3d_head",
            texture_png=tex_png,
            # Split the eyeballs into their own primitive so they can be wet
            # and glossy while the skin stays matte.
            eye_mask=self.eye_mask,
            # Own primitive so a consumer can restyle or hide the hair without
            # touching the head, and so it can be double-sided.
            hair_mask=self.hair_faces if hair_off is not None else None,
        )
        return gltf, blob
