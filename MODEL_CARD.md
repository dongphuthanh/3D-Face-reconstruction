# Model card — face3d encoder

Photograph → FLAME parameters → rigged glTF head.

## The checkpoint

**`runs/deca_open`** is the model. One checkpoint ships; there is nothing to
choose between.

| | |
|---|---|
| FLAME basis | FLAME 2023 Open (CC-BY-4.0) |
| Training corpora | DigiFace-1M + FFHQ, permissive subset |
| Encoder | ResNet-50, ImageNet-pretrained |
| Eyes | predicted — iris colour tracks the photograph at r = +0.822 over 80 faces |
| Redistributable | yes, with attribution to FLAME |

It is licence-clean end to end: a CC-BY basis, a corpus that permits
redistribution of trained models, and no DECA-derived weights anywhere. Earlier
checkpoints in this project's history were trained against FLAME 2020, whose
licence forbids redistribution, and some against CelebA, which forbids
commercial use of derived data. None of them ship, and the comparisons that
retired them are in [FINDINGS.md](FINDINGS.md).

## Inference

```
SHAPE_CALIBRATION = 0.40      # face3d/learn/encoder.py
```

Non-negotiable, and not a tuning nicety. **The raw prediction is worse than
emitting a constant mesh** — 1.5485 mm against the mean face's 1.3554 on the
FLAME 2020 checkpoints, where the full α sweep was run. Discarding 60% of the
predicted shape deviation is the only reason the model beats the mean face.
`deca_open`'s calibration was re-measured on its own basis rather than assumed,
and lands at the same 0.40. `predict(calibrate=True)` applies it; anything
calling `predict()` without it ships an over-confident face.

Crop at **margin 1.6** with MediaPipe landmarks. Every corpus and the NoW eval
use 1.6, and a mismatch is a distribution shift that has bitten this project
before (eval cam scale 9.2 against train 6.9).

## What it does well, and what it does not

Works: pose and camera (holds through ±40° yaw and on profile inputs),
albedo and skin tone, expression and the eyelid/lip terms, alignment tight
enough that a render composites seamlessly into the source photograph.

Weak: **identity shape.** Measured over 200 faces, the shipped output sits
2.18 mm from the mean face while two different people's outputs differ by only
1.79 mm — against a ground-truth error of 1.24 mm, so signal is ~1.4x noise.
Rendered as neutral geometry, a toddler and an elderly woman come out visibly
similar (`out/mean_face_check.png`).

Context, not excuse. All figures below are the NoW **non-metrical**
(scale-invariant) protocol, which is what `face3d.now.run_docker_eval` runs —
`compute_error.py` defaults to `metrical_eval=False` and we never pass
`--metrical_evaluation`. Mixing the two leaderboards is an easy way to quote a
number that means something else.

| | NoW non-metrical median |
|---|---|
| FLAME mean face (ignores the photo) | 1.3693 |
| **this pipeline** | **1.2798** |
| DECA | 1.09 |
| MICA (best published) | 0.98 |

The whole achievable band is therefore ~0.39 mm wide. This pipeline has taken
~23% of it, or ~32% of the distance to DECA. The mean-face baseline is measured
on the same FLAME 2023 Open basis the model uses; against FLAME 2020's 1.3554
the numbers are not comparable, because the mean face is a different mesh. Monocular identity shape is
genuinely underdetermined and every published number sits close to the constant
baseline.

MICA's 0.98 comes from **2,315 subjects with real 3D scan supervision**
(LYHM 1211, FRGC 531, FaceWarehouse 150, Stirling 133, BP4D+ 127, BU-3DFE 100,
Florence 53, D3DFACS 10), against DECA's ~2M images with none. Three orders of
magnitude less data, a better result. That is the strongest available evidence
that the bottleneck here is the *kind* of supervision, not its quantity — and
it is consistent with six corpus experiments in this project all landing inside
1.24–1.29.

Cannot represent: eyewear, facial hair, hair, tongue, ears in detail. FLAME has
no basis for them. An opt-in scalp shell exists but is off — see Export.

## Input quality dominates everything

Measured on three photographs of one person:

| input | recognisability | geometry gain over mean face |
|---|---|---|
| studio portrait, white bg, no glasses, 700px | **0.715** | +0.226 |
| outdoor, harsh side light, 400px | 0.529 | +0.159 |
| glasses, banner overlay, 400px | 0.396 | +0.107 |

Population average over 61 held-out subjects is 0.489. **The spread between a
good and a bad photograph of the same person is larger than the spread between
any two models trained in this project.** A capture guide — face the camera,
even lighting, no glasses, plain background, fill the frame — is worth more
than further training.

Averaging shape across multiple photographs does **not** help (0.525 → 0.477).
Per-image error is systematic bias, not independent noise, so averaging mixes
biases in rather than cancelling them. Pick the best photograph instead.

## Licensing

Weights are our copyright; what constrains them is what they were built from —
and that includes the FLAME basis, not just the image corpora.

| asset | terms | trained models redistributable |
|---|---|---|
| FLAME 2023 Open | CC-BY-4.0 | yes, with attribution |
| DigiFace-1M | R-UDA v1.0 | **yes, explicitly** |
| FFHQ (permissive subset) | CC BY / PD / CC0 | yes |
| MediaPipe segmenter / landmarker | Apache-2.0 | yes |

That is the whole list, and it is why one checkpoint ships. Exported meshes may
be redistributed with attribution to FLAME. No DECA-derived weights are used
anywhere — where the code says "DECA weights" it means their published *loss
weights*, read from their config, not their model.

Two constraints that shaped this and are worth stating because they are easy to
trip over:

**A FLAME 2020 checkpoint cannot be repointed at the 2023 Open basis.** The two
share topology (5023 verts, 9976 faces, byte-identical `faces`) and are equally
expressive — the first 100 components hold 99.20% of shape variance in 2020
against 98.87% in 2023 Open. But their axes are rotated relative to each other:
expressing either basis's top-100 directions in the other's span retains 81% of
the energy, in both directions. The coefficients mean different things, so a
2020-trained encoder emits nonsense through a 2023 basis. Accuracy is not the
cost — 2023 Open is not the weaker basis — the cost is one retraining run, which
is what produced this checkpoint.

**`face3d/assets.py` used to resolve FLAME implicitly**, listing 2020 first in
`CANDIDATES`, so it silently won over the `FLAME2023Open` copy beside it. That
is how a project can believe it is shipping a clean basis while every exported
GLB carries a non-redistributable one. Pass the basis explicitly.

This project is non-commercial. Not legal advice; get real advice before any
commercial use.

## Reproducing

```bash
FACE3D_MODEL=FLAME2023Open/flame2023_Open.pkl python scripts/train/train.py \
  --identity-data data/digiface --epochs 3 --batch 8 -k 4 --lr 2e-4 \
  --w-swap 1.0 --w-id 0.2 --w-eye 1.0 --w-lip 0.5 \
  --mix-ffhq 1.0 --mix-batch 16 --out runs/deca_open
```

`FACE3D_MODEL` is not optional. Without it `face3d/assets.py` resolves FLAME
implicitly and picks up whichever basis it finds first.

Evaluation: `scripts/eval/now_predict.py` + Docker
(`face3d.export.now.run_docker_eval`), then `scripts/eval/eval_closure.py`,
`scripts/eval/diag_identity.py`, `scripts/tools/render_compare.py`.

Export: `scripts/tools/export_head.py --image X --out head.glb` — defaults to
the `deca_open` / FLAME 2023 Open pair; pass `--checkpoint` and `--flame-model`
together or not at all.

## Export

GLB carries: **5118 verts** (5023 geometry vertices with 95 duplicated along UV
seams -- see below), 9976 tris in two primitives (skin, eyes) sharing one set of vertex accessors, smooth normals, UVs, an embedded 512×512
baseColour PNG, a 5-joint armature
(root/neck/jaw/eye_left/eye_right), and 21 named morph targets
(`expr_00`…`expr_19`, `jaw_open`). Head is ~31 cm tall at the origin, Y-up.
Validator-clean (0 errors, 0 warnings).

**Seam vertices are duplicated, not collapsed.** FLAME's unwrap needs 5118 UVs
for 5023 vertices, because a seam is a cut where one vertex appears at two
places in the texture; glTF allows one UV per vertex. Collapsing them is wrong
for 284 triangle corners (0.95%), which sit on the inner-mouth seam and sampled
somewhere unrelated -- a bright patch inside the mouth, visible from below. It
appeared only in the exported GLB, never in any render here, because those
sample through the correct per-corner UVs. Normals are still computed on the
UNSPLIT mesh: recomputing after the split would treat the seam as a boundary
and shade it as a crease.

The texture is **sampled from the photograph**, not reconstructed from the 50
albedo coefficients (`face3d/texture/project.py`). The basis cannot represent a mole, a
freckle, an eyebrow or stubble, because it holds no direction that varies at
that scale; projecting the photo through the fitted camera recovers all of
them, and eyebrows in particular read convincingly as texture on geometry that
has none. The basis still fills whatever the camera never saw, mirrored across
the face where a mirror is available. Adds ~250 ms. `--no-project` restores the
old PCA-only bake.

**A learned generator is the default** (`face3d/texture/texgen.py`,
`runs/texgen_gen`). Projection still exists behind `texgen=False`, and its two
guards are described below, but a guard is a filter and a filter can only
remove what it recognises. The generator instead makes the failure
unreachable: the texture is reconstructed from a 128-vector by a decoder
trained only on faces, so a pair of spectacle frames is not in its range, and
the output is a residual over the PCA basis bounded to +/-0.45, so no texel can
move far from something the basis already considered plausible. The worst
failure available to it is looking generic.

Trained on 5,738 gated projections from FFHQ, in two stages -- an autoencoder
first, to establish that a latent this small can hold these textures at all,
then a photograph-to-latent head warm-started from its decoder.

| masked L1 vs the photograph, 286 held out | |
|---|---|
| PCA basis | 0.1508 |
| generator + procedural layer (ships) | **0.0564** — 62.6% closer |
| generator alone, no named parameters | 0.0488 — 67.4% closer |
| autoencoder (upper bound, sees the target) | ~0.039 — 74% closer |

The procedural layer costs ~7 points of pixel fidelity and buys editability —
see below. Judge it on renders: this metric rewards matching pixels and cannot
see a slider.

The correction is **continuous over the whole head**, and this is the single
thing that most changed how the output reads. Supervising the residual to be
zero outside the observed region is a defensible safety default and also an
instruction to paint a face-shaped patch: measured on the first trained model,
mean |residual| was 0.1275 inside the face mask against 0.0043 outside, a 30x
step, and a step in the correction is a visible edge on the head. It looked
like a mask laid over a mannequin. The target is now the measurement where
there is one and its own smooth continuation where there is not, so the neck
and jaw inherit the same skin the face got. The step ratio is 1.0x.

The generator sits close to the autoencoder's ceiling, so predicting the latent
from a photograph costs little against having the texture itself: the
bottleneck is the corpus and the representation, not the encoder. Rendered, it
keeps wrinkles, age and skin tone while dropping the beaded headdress, the
spectacle frames and the hair blown across a cheek that projection transfers
faithfully. It is also the fastest path -- no projection, no segmentation --
at 0.27 s through the HTTP layer.

## Named skin parameters

Eleven numbers on top of the learned texture (`face3d/texture/skin.py`), predicted from
the same trunk: **skin RGB, lip RGB, brow RGB, freckle amount, crease amount**.
`Reconstructor.build(..., skin_override={...})` replaces any of them without
touching the rest, and `last_params` reports what the photograph implied. This
is the character-creator half: a curated parametric space where every setting
is valid by construction, rather than a fit that happens to land somewhere
plausible.

The colour parameters are supervised directly against the mean colour of the
region they are named after, which is what makes them mean what they are
called. Trained only through image error they would drift to whatever value
reduced the loss, and "lip colour" would stop being lip colour -- useless as a
slider.

Regions come from the MediaPipe landmark embedding already on disk: all 40 lip
and 20 brow landmarks are among the 105 embedded, so those regions are exact.
The nose is only half covered and so has no region. The crease map is derived
from the corpus rather than authored -- averaging the darkening half of the
high-frequency residual over 800 faces leaves where creases land in most people
and averages away what was specific to any one of them.

REGIONAL OPERATIONS ARE THE HAZARD. Skin tone is measured on the skin region
and applied to the WHOLE head; confining the shift to the region would draw its
boundary, which is the same failure as the polygon edge across the forehead and
the face-shaped patch. Only lips and brows are regional, and both fade over
several millimetres.

The freckle field is a fixed noise image, never re-randomised: the same
photograph must not produce a different face on each upload.

## Trade-offs

It is blurrier than projection at its best, which is the trade taken
deliberately: sacrifice similarity, never produce something weird. Val bottomed
around epoch 33 and flattened while train kept falling, so more data would help
before more capacity would.

Because the correction now covers the whole head, a strong colour cast in the
photograph tints the scalp and neck as well as the face. That is the intended
behaviour -- it is what makes the head one object -- but on a photo lit by, say,
foliage it reads as a tinted head rather than a tinted face.

Regions no frontal camera could have seen -- under the jaw, round the back --
are settled toward the skin tone measured on this person. Left alone the
underside of the jaw reached 1.68x the face's mean brightness on the worst of 30
subjects and saturated to near-white, from two causes that compound: the PCA
basis is itself bright there (1.29x against 0.97x for others, because the
texture space was built from photographs and almost nobody photographs under a
jaw), and diffuse_fill then extends the face's correction there additively with
no knowledge that the base has no headroom left. Now max 1.21x, median 1.10x.
The mask is by surface ORIENTATION, not by the face polygon -- face_texel_mask
includes the under-jaw, so keying it there applied at a quarter strength.

`harmonise()` is skipped under the generator. It fills everything outside the
face mask with the fitted face's mean tone, which is a hand-made approximation
of what the generator now learns; running both would flatten the extension back
to a constant and reinstate the boundary it exists to remove.

---

The projection path, still available with `texgen=False`. Two guards, because
raw projection transfers whatever the photograph contains, faithfully, onto a
head that cannot represent it.

**Occluders are gated out by segmentation.** Audited over 14 FFHQ faces, 8
carried one — spectacle frames, a fringe across the forehead, a beaded
headdress, hair blown over a cheek — and this, not smearing, was the dominant
artefact. MediaPipe's selfie multiclass model marks face skin, and only face
skin may be sampled. The class boundary falls exactly where it is needed:
eyebrows and lips are face skin and survive; glasses and headdresses are
`other`, hair over the forehead is `hair`, and neither does.

**Tone comes from the basis, detail from the photograph.** The two sources fail
in opposite bands. The basis has no high frequencies at all, but its low
frequencies cannot invent a dark band across a forehead. The projection is the
reverse: its detail is a real measurement, while its low frequencies carry every
error left unfixed — cast shadows order-2 SH has no model for, specular
highlights, a room's colour cast, the broad smear where geometry is wrong. Those
are large, soft and wrong, which is the most visible combination. So the
projection's detail is kept and the basis sets the level under it.

What still gets through: a cast shadow hard enough to read as an edge, and an
occluder the segmenter calls skin. Where geometry is wrong the texture still
smears — which is what makes the capture guide above load-bearing rather than
cosmetic.

**There is no hair.** The exported head is bald, and FLAME has no basis for
hair, so this is a limitation of the representation rather than a missing
feature.

A scalp shell was built and removed. It segmented the photograph's hair, compared
the hair outline to the scalp outline per angle around the head, pushed the
scalp out along its normals by the difference, and let the projection paint the
person's own hair onto it. The measurement worked. The result read *worse* than
a bald head: it lands in the valley between "no hair" and "that person's hair",
where a viewer reads the near-miss as wrong rather than the absence as stylised.
It could not do a parting, a curl, a ponytail, or anything leaving the skull.
Kept behind a flag for a while, then deleted — a disabled feature is still code
to read, and the fix is real hair geometry (authored assets or strand
reconstruction), not a better fit to a silhouette.

The segmenter it used survives as `face3d/texture/segment.py`, because gating
occluders out of the projection needs it regardless.

## Privacy

Face images are biometric data under GDPR and BIPA. This project's rules: no
scraping, consent on file for every demo photograph, and licence-register rows
for every corpus. Anything that accepts uploads should not retain them.

Since the texture became a projection, **an exported GLB contains the
photograph itself**, wrapped onto a mesh. That is a different object from 50
PCA coefficients: the earlier bake could not reproduce a recognisable image of
anybody, and this one is a recognisable image by construction. "Do not retain
uploads" is satisfied in `webapp/pipeline.py`, which never writes one to disk —
but the file the user downloads now carries their face as pixels, and any
sharing of it is sharing the photograph.
