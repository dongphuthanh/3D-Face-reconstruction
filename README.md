# face3d

Monocular 3D face reconstruction: one photo in, a rigged glTF head asset out.

**Status: early.** The differentiable core is built and tested. The asset
pipeline (rigging, glTF export) and the web frontend are not started. Numbers
below are measured on this machine, not quoted from papers.

---

## What works today

A complete self-supervised training loop, running end to end on one consumer GPU:

```
photo → ResNet-50 → FlameParams → FLAME → rasteriser → losses → ∂/∂θ
        └─ trainable ─┘           └────────── frozen ──────────┘
```

FLAME is a fixed statistical basis. Gradients flow *through* it to the encoder;
its own arrays are registered as buffers, so `model.parameters()` is empty and no
optimiser can touch the basis by accident.

| Measurement | Value |
|---|---|
| FLAME forward (batch 32) | 16,000 meshes/s, 66 MB VRAM |
| Rasteriser (batch 8, 224px) | 846 img/s, 214 MB VRAM |
| Training step (batch 12, 224px) | ~0.9 s, 2.72 GB peak (paired views) |
| Torch↔NumPy parity | 8.3e-17 m |
| Overfit a single batch (C1) | 3.73 mm mean vertex error, 83× loss drop |

Hardware: RTX 5070 Laptop (Blackwell sm_120, 8 GB), torch 2.9.0+cu128.

## What is measured, and what it cost to learn

**Pretrained initialisation is not optional.** Identical seeds, data and losses:

| Backbone | Loss drop | Vertex error | Jaw error |
|---|---|---|---|
| Random init | 26× | 9.24 mm | 0.519 rad (~30°) |
| ImageNet-pretrained | 83× | 3.73 mm | 0.008 rad |

The random-init failure is deceptive: renders match the targets visually and the
loss falls steadily. It looks like a bad loss weighting. It is not.

**Landmarks dominate shading.** Fitting parameters from zero, synthetic ground
truth, same renderer:

| Objective | Vertex error |
|---|---|
| Geometric (3D vertices) | 0.12 mm |
| Image only (shading) | 18.9 mm |
| Image + sparse 2D landmarks | 2.9 mm |

All three produce renders that look like plausible faces. The image-only fit is
visually convincing and geometrically wrong by 19 mm. **Visual plausibility is
not evidence of accuracy** — which is why the NoW benchmark, not eyeballing,
decides whether this works.

**Scaling identity-grouped data beats every loss change.** 10,000 DigiFace
identities (60,000 images) with the shape swap and an FFHQ mix:

| alpha | 0.00 | 0.25 | **0.50** | 0.75 | 1.00 |
|---|---|---|---|---|---|
| id_swap, 2k identities | 1.3554 | **1.3300** | 1.3485 | 1.4214 | 1.5296 mm |
| digi_all, 10k identities | 1.3554 | 1.3294 | **1.3241** | 1.3602 | 1.4242 mm |

Two things moved. The best score improved to **1.3241 mm** against a mean-face
baseline of 1.3554, and the optimum shifted from alpha 0.25 to 0.50 -- twice as
much of the prediction became usable, which is calibration improving rather
than just a better number. Identity ratio on NoW rose 0.68 -> 0.75, the highest
measured, while deviation fell to 2.42 mm: more identity-correlated AND less
over-confident, where every earlier gain traded one for the other.

Worth noting what did NOT work. Arc2Face -- 10k identities of real 448px
photographs, against DigiFace's synthetic 112px -- reached a better in-domain
ratio (1.06 vs 0.72) and a worse NoW ratio (0.63 vs 0.75), with no dip at any
alpha. Neither the FFHQ mix nor DECA's photometric swap changed that.

**Real identity pairs produce the first genuine identity signal.** DECA's
shape swap needs two images of the same person; augmented views of one image
only ever taught invariance to the augmentation. DigiFace-1M supplies real
pairs (2,000 synthetic subjects, 6 renders each, varying pose/expression/light).

Shrinkage sweep -- scaling predicted shape from the FLAME mean (0) to the full
prediction (1):

| alpha | 0.00 | **0.25** | 0.50 | 0.75 | 1.00 | 1.25 |
|---|---|---|---|---|---|---|
| NoW median | 1.3554 | **1.3300** | 1.3485 | 1.4214 | 1.5296 | 1.6648 mm |

**A dip, and it goes below the mean-face baseline.** Every earlier sweep was
strictly monotonic -- the predicted shape was always worth discarding. This one
says keep a quarter of it. The direction is right and the magnitude is roughly
4x too large: the encoder is over-confident, which is a calibration problem
rather than an absence of signal.

Corroborating: id_swap predicts MORE deviation than the 20k baseline (3.23 mm
vs 3.03) and scores BETTER (1.530 vs 1.614). Noise added to a good prior cannot
do that; only deviation correlated with the truth can. Every previous
"improvement" had the opposite fingerprint -- less deviation, drifting toward
the mean.

Against its matched control (same data, swap off) the swap is worth 11% on NoW
(1.710 -> 1.530) and moves the in-domain ratio 0.67 -> 0.72.

**Data moved identity more than any loss change.** Quadrupling the corpus
from 4,996 to 19,978 images, with no other change:

| configuration | corpus | NoW median | identity ratio |
|---|---|---|---|
| DECA (test split) | ~2M, identity-grouped | ~1.09 mm | - |
| **FLAME mean face** | - | **1.355 mm** | - |
| baseline | 20k | 1.614 mm | 0.65 |
| + C2 skin mask | 5k | 1.701 mm | 0.41 |
| + DECA-style shape swap | 20k | 1.741 mm | 0.60 |
| baseline | 5k | 1.841 mm | 0.42 |

**Landmarks are a pose signal, not a shape signal.** Replacing one parameter
group with another sample's value and measuring how far the 105 projected
landmarks move:

| swapped group | pose | camera | shape | expression |
|---|---|---|---|---|
| landmark shift | **17.13 px** | 3.70 px | **1.95 px** | 1.41 px |

Pose moves the landmarks nearly 9x more than shape does, yet the landmark term
carries weight 5.0 against the photometric term's 1.0. The objective is
dominated by the one loss that is nearly blind to identity, which is why the
DECA-style swap changed nothing when routed through it -- swapping shape can
only perturb that loss by about 1.4%. The swap needs the photometric term,
where shape has real leverage.

**The shape channel carries no identity signal.** Scaling predicted shape
by alpha interpolates from the FLAME mean face (0) to the full prediction (1):

| alpha | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| baseline, 5k | **1.355** | 1.417 | 1.507 | 1.651 | 1.841 mm |
| + C2 skin mask, 5k | **1.355** | 1.376 | 1.430 | 1.546 | 1.701 mm |
| baseline, 20k | **1.355** | 1.364 | 1.401 | 1.489 | 1.614 mm |

The initial slope -- what a unit of predicted shape costs -- fell from
0.244 mm per unit alpha at 5k to **0.033 mm at 20k**, a factor of 7.4. The
deviation is still noise, but noise an order of magnitude closer to parity;
a dip is what parity becoming benefit would look like.

Both strictly monotonic, with no optimum above zero. Any genuine identity
signal -- however weak -- would produce a dip at some alpha. There is none, so
the learned deviation is indistinguishable from noise added to a good prior,
and the mean face wins because uninformative deviation strictly increases
expected error.

The skin-mask curve sits below the baseline at every alpha: same shrinkage
behaviour, gentler slope. C2 did not teach the encoder who anyone is, it made
its errors less harmful per unit of magnitude. NoW alone called that a 7.6%
improvement and the identity ratio called it unchanged (0.42 -> 0.41); both are
true, and only the sweep distinguishes them.

`scripts/visualize_predictions.py` shows the same thing directly: five subjects,
five photographs each, every prediction reduced to its neutral shape under one
fixed camera produces the same generic face. Meanwhile pose, expression,
lighting and albedo all transfer correctly -- five of the encoder's six output
groups work, and the inert one is the only thing NoW measures, since the metric
aligns away camera and pose and scores against a neutral scan.

**A single metric misleads.** Two 8-epoch runs, identical but for a
shape-consistency term across augmented views:

| configuration | NoW median | identity ratio |
|---|---|---|
| published DECA (test split) | ~1.09 mm | - |
| **FLAME mean face** | **1.355 mm** | - |
| + consistency (w=0.05) | 1.560 mm | 0.29 |
| baseline | 1.841 mm | 0.42 |

Consistency improved NoW by 15% and made identity learning *worse*. It cut
within-subject shape spread 3x, as designed -- but cut between-subject spread
4.4x, so the encoder became invariant to identity along with everything else.
`(shape_A - shape_B)^2` is minimised perfectly by predicting a constant, and
nothing in the objective rewards shape varying between people. The NoW gain is
regression toward the mean face, which outscores both runs.

The fix is an explicit anti-collapse term (VICReg-style variance and covariance
regularisation), not a different weight. `scripts/diag_identity.py` reports the
between/within ratio precisely because the millimetre figure alone would have
called this a success.

**Landmarks and albedo are substitutes.** Direct parameter optimisation against
a textured synthetic target, 1200 iterations, only the fitted model varying
(`scripts/ablation.py`):

| | no landmarks | real landmarks |
|---|---|---|
| grey albedo | 30.14 mm | 5.45 mm |
| texture albedo | 6.18 mm | 5.43 mm |

Either signal alone rescues the fit; both together add almost nothing. Albedo is
what makes the photometric term work at all — 4.9x on its own — but buys no extra
geometric accuracy once landmarks are present. Its value is as an *independent*
path to a fit, which is what matters when landmarks fail on profiles and
occlusions.

Caveats in both directions: the target is generated by the same texture basis
being fitted, which flatters texture; and the landmarks are read from ground
truth with no detector noise, which flatters landmarks.

**Four bugs found, all the same species: plausible output, broken internals.**

1. `torch.where(θ < ε, I, R)` in Rodrigues returns the right rotation at zero
   pose but a *constant*, so autograd sends zero gradient. Every training run
   initialises at zero pose — jaw, neck and head rotation would never have
   learned.
2. Fixing (1) naively produced NaN: `torch.where` evaluates the discarded branch
   anyway, and `inf × 0 = NaN`. Every denominator needs the masked value.
3. A zero-initialised regression head makes `∂L/∂trunk = ∂L/∂out · W` identically
   zero, starving the entire backbone.
4. The FLAME texture space stores channels BGR. Rendered faces came out blue —
   structurally perfect, wrong colour. The photometric loss would have absorbed
   the error into distorted albedo coefficients rather than failing.

None would have been caught by looking at outputs. Hence tests that assert on
invariants — gradients non-zero at initialisation, skin satisfying R > G > B —
rather than on whether a render looks like a face.

## Layout

```
face3d/
  flame_np.py     NumPy FLAME — the oracle the torch port is diffed against
  flame_torch.py  Differentiable FLAME, batched, CUDA
  render.py       Pure-torch differentiable rasteriser + SH shading
  params.py       FlameParams: the typed encoder↔pipeline contract
  encoder.py      Encoder protocol + ResNet-50 regressor
  pipeline.py     params → mesh → projected landmarks → shaded image
  losses.py       landmark, photometric, identity, shape-consistency, regularisation
  assets.py       Locates a FLAME model; lets tests skip when it is absent
scripts/          One runnable check per concern; run_tests.py runs them all
```

## Running it

```bash
make setup     # pinned deps, CUDA 12.8 build of torch
make test      # full suite against whichever FLAME model is on disk
make test-ci   # same suite against the synthetic fixture, no licensed assets
```

`make setup` **must** use the cu128 index on Blackwell GPUs. Older CUDA wheels
install cleanly and then fail at the first kernel launch.

### Rasteriser

`face3d/render.py` is a hand-written differentiable rasteriser in plain PyTorch,
not nvdiffrast — which needs `nvcc`, MSVC and `ninja` to build and cannot compile
on the development machine. It mirrors nvdiffrast's interface (`rasterize` returns
face ids plus differentiable barycentrics; `interpolate` samples vertex
attributes), so swapping in the real thing is a module change. Like nvdiffrast
without antialiasing, gradients are correct inside triangles and absent across
occlusion boundaries.

### Why CI runs on a fake model

No FLAME variant may be committed — 2020 and 2023 are non-commercial and
non-redistributable, and even CC-BY 2023 Open is 51 MB. `scripts/make_fixture.py`
generates a model sharing FLAME's pickle structure from an icosphere and random
bases, containing no MPI data. It exercises LBS, the joint hierarchy,
blendshapes, rasterisation, gradients and the encoder interface at 162 vertices.
Suites whose thresholds are stated in millimetres against a real head skip
themselves.

## Assets — not included

Every asset is licensed and must be obtained separately. See §9 of the
implementation plan for the full register.

| Asset | Source | Redistributable |
|---|---|---|
| FLAME 2020 / 2023 | flame.is.tue.mpg.de | No |
| FLAME 2023 Open | same | Yes (CC-BY-4.0) |
| NoW benchmark | now.is.tue.mpg.de | No |
| DECA weights | DECA repo | **No** |
| FFHQ (permissive subset) | NVlabs | Yes (CC BY / PD / CC0) |
| DigiFace-1M | Microsoft | Data no; trained models yes (R-UDA) |
| Arc2Face | HuggingFace | CC BY-NC-SA 4.0, ShareAlike |

Arc2Face is derived from WebFace42M, i.e. scraped source images. The CC licence
covers the authors' restoration work, not the provenance of what was restored,
and ShareAlike may bind models trained on it. It is used because 448px real
photographs with 30k identities per archive is what the identity signal needs;
the trade against the project's no-scraping rule is deliberate and recorded
here rather than glossed.

**This project is non-commercial.** DECA's licence forbids distributing the
model, so shipping DECA-derived weights to a browser is not possible; on-device
inference is gated on training an encoder against FLAME 2023 Open.

Still needed before training on real photographs: the FLAME landmark embedding,
a 2D landmark detector, the BFM albedo basis, a face-parsing network, and an
identity-labelled image corpus.

## Evaluation

The official `now_evaluation` metric runs through the Dockerfile it ships
(`make eval-image`); it pins numpy 1.19.5 and chumpy 0.70 against Python <=3.9
and needs a Unix C++ toolchain, so it will not install natively on Windows.

Harness validated with two baselines that need no trained model (`make eval-check`):

| Baseline | Median | Mean |
|---|---|---|
| Identity (ground truth as its own prediction) | 3.1e-07 mm | 5.9e-07 mm |
| FLAME mean face, 20 validation subjects | **1.355 mm** | 1.703 mm |

The identity run confirms the metric pipeline is correct end to end. The mean-face
run is the number that matters for planning: **predicting the average face scores
1.355 mm**, against roughly 1.09 mm for published DECA. The entire headroom
between predicting nothing and the state of the art is about 0.27 mm, so any
claimed improvement needs statistics to match.

NoW aligns predictions using 7 landmarks and ships only a picture of where they
are. They are recovered from the MediaPipe embedding and validated by Procrustes
against all 20 ground-truth landmark files: 7.44 mm RMS for the correct ordering
versus 19.00 mm mirrored and 35.67 mm for random permutations.

Use the **non-metrical** protocol: published DECA numbers are non-metrical, and
comparing against the metrical leaderboard produces a large unexplained gap.
