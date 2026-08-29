# Model card — face3d encoder

Photograph → FLAME parameters → rigged glTF head.

## Which checkpoint to ship

**`runs/deca_open` unless you have a reason not to.** It is the only
licence-clean checkpoint, the only one with predicted eyes, and it gives up
nothing measurable to do it. `deca_joint` remains marginally ahead on expression
if the work is non-commercial and stays private.

| | deca_full | deca_celswap | deca_joint | **deca_open** |
|---|---|---|---|---|
| FLAME basis | 2020 | 2020 | 2020 | **2023 Open** |
| mean-face baseline | 1.3554 | 1.3554 | 1.3554 | **1.3693** |
| NoW median (best α) | 1.2679 | 1.2408 | 1.2688 | 1.2798 |
| **gain over own baseline** | 0.0875 | **0.1146** | 0.0866 | 0.0895 |
| closure, CelebA held-out (real) | 6.2 / 13.4% | 5.2 / 12.2% | **5.0 / 11.1%** | — |
| closure, DigiFace held-out | 6.2 / 11.8% | 7.5 / 16.2% | 6.3 / 11.9% | **6.6 / 11.6%** |
| identity ratio | 1.15 | 1.14 | **1.16** | 1.14 |
| iris tracks the photo | no | no | no | **r = +0.822** |
| licence-clean weights | no | no | no | **yes** |

**Compare the gain row, not the median row.** deca_open sits on a different
basis whose mean face is a different mesh (1.3693 against 1.3554), so its
median is not commensurable with the others. Against its own baseline it takes
marginally more of the available distance than deca_full does.

`deca_celswap` wins NoW by 0.028 mm. `deca_joint` wins everything that scores
what a viewer actually sees — expression fidelity on both domains, and
recognisability. NoW scores a *neutral* mesh, so it is an identity-shape metric
and blind to expression by construction (`now_predict.py --neutral` zeroes
expression and pose). For a rigged head, where the neutral mesh is the base and
blendshapes carry expression, the metrics that favour `deca_joint` are the ones
aligned with the product.

Seed variance has never been measured. A 0.028 mm gap is below what this setup
can resolve, so treat the NoW ordering as a tie.

## Inference

```
SHAPE_CALIBRATION = 0.40      # face3d/encoder.py
```

Non-negotiable, and not a tuning nicety. The raw prediction (α=1.0) scores
**1.5485 mm on NoW — worse than emitting the FLAME mean face (1.3554)**.
Discarding 60% of the predicted shape deviation is the only reason the model
beats a constant mesh. `predict(calibrate=True)` applies it; anything calling
`predict()` without it ships an over-confident face.

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
| FLAME mean face (ignores the photo) | 1.3554 |
| **this pipeline** | **1.2408** |
| DECA | 1.09 |
| MICA (best published) | 0.98 |

The whole achievable band is therefore ~0.375 mm wide. This pipeline has taken
~31% of it, or ~43% of the distance to DECA. Monocular identity shape is
genuinely underdetermined and every published number sits close to the constant
baseline.

MICA's 0.98 comes from **2,315 subjects with real 3D scan supervision**
(LYHM 1211, FRGC 531, FaceWarehouse 150, Stirling 133, BP4D+ 127, BU-3DFE 100,
Florence 53, D3DFACS 10), against DECA's ~2M images with none. Three orders of
magnitude less data, a better result. That is the strongest available evidence
that the bottleneck here is the *kind* of supervision, not its quantity — and
it is consistent with six corpus experiments in this project all landing inside
1.24–1.29.

Hair is a fitted cap, not a hairstyle — see Export below.

Cannot represent: eyewear, facial hair, tongue, ears in detail. FLAME has no
basis for them.

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
| **FLAME 2020** | non-commercial research | **no** |
| FLAME 2023 Open | CC-BY-4.0 | yes, with attribution |
| DigiFace-1M | R-UDA v1.0 | **yes, explicitly** |
| FFHQ (permissive subset) | CC BY / PD / CC0 | yes |
| CelebA | non-commercial, no redistribution of "derived data" | **no** |
| MediaPipe segmenter / landmarker | Apache-2.0 | yes |

**`runs/deca_open` is licence-clean.** FLAME 2023 Open (CC-BY-4.0) with
DigiFace + FFHQ only — no CelebA, no FLAME 2020. It is the model to use for
anything public, and the only one whose exported meshes may be redistributed
(with attribution to FLAME).

Its NoW median is 1.2798, which must NOT be read against the table above: the
mean face is a different mesh, measured at 1.3693 mm against FLAME 2020's
1.3554. Compare gains — deca_full takes 0.0875 of its baseline, deca_open
0.0895 of its own, so the clean basis costs nothing. Identity ratio 1.14 and
closure 6.6 / 11.6% are ties with deca_full. SHAPE_CALIBRATION stays 0.40,
re-measured rather than assumed.

It is also the only model with predicted eyes: iris colour correlates with the
photographed iris at r = +0.822 over 80 faces. Every other checkpoint emits the
albedo basis mean's blurred brown iris for every subject.

**The older checkpoints are not licence-clean, `deca_full` included.** Every run was
trained against `FLAME2020/generic_model.pkl`. `face3d/assets.py` lists FLAME
2020 first in `CANDIDATES`, so it wins over the `FLAME2023Open` copy sitting
beside it, silently. Every GLB this project has exported is FLAME 2020 geometry.

Separately, `deca_joint` and `deca_celswap` use CelebA, whose agreement forbids
exploiting "any portion of derived data" commercially. `deca_full` avoids that
one — DigiFace + FFHQ only — so it is the *corpus*-clean option, but the FLAME
2020 dependency binds it equally.

Getting to genuinely clean weights means retraining against FLAME 2023 Open:

```bash
FACE3D_MODEL=FLAME2023Open/flame2023_Open.pkl python scripts/train.py ...
```

An existing checkpoint cannot simply be repointed. The two bases share topology
(5023 verts, 9976 faces, byte-identical `faces`) and are equally expressive —
first 100 components hold 99.20% of shape variance in 2020 against 98.87% in
2023 Open, and 2023 Open's leading component is slightly *larger* (3.815 mm
against 3.585 mm at 1σ). But their axes are rotated relative to each other:
expressing either basis's top-100 directions in the other's span retains 81% of
the energy, in both directions. The coefficients mean different things, so a
2020-trained encoder emits nonsense through a 2023 basis.

The good news is that accuracy is not the cost — 2023 Open is not the weaker
basis. The cost is one retraining run.

This project is non-commercial, so nothing is currently violated.

Not legal advice. Get real advice before any commercial use.

## Reproducing

```bash
# deca_joint
python scripts/train.py --identity-data data/digiface \
  --identity-data2 data/celeba --identity-cache2 landmarks_224_swap.npz \
  --identity-min-images2 4 --identity2-batch 8 --w-identity2 1.0 \
  --epochs 3 --batch 8 -k 4 --lr 2e-4 \
  --w-swap 1.0 --w-id 0.2 --w-eye 1.0 --w-lip 0.5 \
  --mix-ffhq 1.0 --mix-batch 16 --out runs/deca_joint

# deca_full (licence-clean)
python scripts/train.py --identity-data data/digiface --epochs 3 --batch 8 -k 4 \
  --lr 2e-4 --w-swap 1.0 --w-id 0.2 --w-eye 1.0 --w-lip 0.5 \
  --mix-ffhq 1.0 --mix-batch 16 --out runs/deca_full
```

Evaluation: `scripts/now_predict.py` + Docker (`face3d.now.run_docker_eval`),
`scripts/eval_closure.py --data {digiface,celeba}`, `scripts/diag_identity.py`,
`scripts/render_compare.py`. Export: `scripts/export_head.py --image X
--out head.glb` (defaults to the deca_open / FLAME 2023 Open pair; pass
`--checkpoint` and `--flame-model` together or not at all).

## Export

GLB carries: 5023 verts, 9976 tris in three primitives (skin, eyes, hair)
sharing one set of vertex accessors, smooth normals, UVs, an embedded 512×512
baseColour PNG, a 5-joint armature
(root/neck/jaw/eye_left/eye_right), and 21 named morph targets
(`expr_00`…`expr_19`, `jaw_open`). Head is ~31 cm tall at the origin, Y-up.
Validator-clean (0 errors, 0 warnings).

The texture is **sampled from the photograph**, not reconstructed from the 50
albedo coefficients (`face3d/project.py`). The basis cannot represent a mole, a
freckle, an eyebrow or stubble, because it holds no direction that varies at
that scale; projecting the photo through the fitted camera recovers all of
them, and eyebrows in particular read convincingly as texture on geometry that
has none. The basis still fills whatever the camera never saw, mirrored across
the face where a mirror is available. Adds ~250 ms. `--no-project` restores the
old PCA-only bake.

It inherits the fit's weaknesses: where geometry is wrong the texture smears,
and cast shadows and specular highlights stay baked in, because order-2 SH has
no model of either. Sunglasses and stray hair project too — which is what makes
the capture guide above load-bearing rather than cosmetic.

**Hair is the scalp inflated to fit the photograph's hair outline**
(`face3d/hair.py`), not a hair model. MediaPipe's selfie multiclass segmenter
(Apache-2.0) marks hair pixels; the hair outline is compared to the scalp
outline per angle around the head; the scalp is pushed out along its normals by
the difference, tapered to nothing at the hairline. The projection then paints
the person's own hair onto it, and unseen scalp is filled with the mean of the
hair that *was* seen.

Be clear what that is. It is a cap following one view's silhouette. It cannot do
a parting, a curl, a strand, or anything that leaves the skull — a ponytail, a
fringe over an eye, hair past the shoulders. For short and medium hair it turns
a bald mannequin into something recognisable; for long hair it gives a helmet
whose outline is right from the front and wrong behind. Thickness is capped at
4 cm, and below 4 mm no shell is emitted at all, so a bald head stays bald
rather than gaining a swollen skull. Doing better means authored hair assets or
strand reconstruction, and neither is a bigger version of this. Adds ~300 ms.
`--no-hair` turns it off.

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
