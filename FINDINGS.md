# Findings

The experiment log behind [README.md](README.md). Every number here was measured
on this machine. Several were wrong on the first measurement and are corrected
in place rather than tidied away, because a result you cannot see the working
for is not worth much.

---

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

**Reading the reference implementation was worth more than any experiment.**
Cloning DECA revealed our landmark:photometric weighting was 5:1 where theirs is
1:2 -- a 10x swing toward the term our own measurement showed is near-blind to
shape -- and that light regularisation, their strongest weight, was missing
entirely. Adopting their configuration (plus K=4 images per identity):

| | identity ratio | best NoW |
|---|---|---|
| 110k identities, our weights | 0.76 | 1.3109 mm |
| **110k identities, DECA weights** | **0.96** | **1.2923 mm** |

The ratio jump (0.76 -> 0.96) is the largest of the project -- larger than 55x
more data (0.68 -> 0.76) or swapping in an ArcFace shape head (0.76 -> 0.85).
It cost nothing but reading their config. Still missing from their setup at
that point: the identity loss (0.2, face recognition on the render),
eye-closure and lip-distance terms, real segmentation masks, and randomised
crop scale.

Four of those five have since been adopted -- the shipped run passes
`--w-id 0.2 --w-eye 1.0 --w-lip 0.5` and `scale_jitter` is applied per
batch. **Real face-parsing masks are the one that remains.** The photometric
mask is still the rendered mesh silhouette restricted to face-skin triangles,
which is mesh-side: it says which triangles to compare, not which PHOTOGRAPH
pixels are actually skin. Glasses, a fringe or a hand falling inside the
projected face still enter the loss, and geometry is the free variable
available to explain them. The segmenter that would fix it already exists
(`face3d/texture/segment.py`) and is used only for the inference-time texture
projection.

**Tripling the texture corpus is the one change that did work.** The generator
was data-limited and said so: train sat 32% below val, and val bottomed around
epoch 33 then turned up. The corpus was 5,738 targets because
`make_texture_corpus.py` defaults to `--limit 6000`, while FFHQ has 19,978
crops -- so 3x more data was available for the cost of compute alone.

Extended to 18,082 targets (`--offset 6000`, so the existing 2.3 GB is reused
rather than rebuilt) and retrained with everything else identical: 45 epochs,
same warm start from `texgen_ae2`.

| | best val | masked L1, 904 held out |
|---|---|---|
| gen6, 5,738 targets | 0.03415 | 0.05899 |
| **gen7, 18,082 targets** | **0.02945** | **0.05644** |

Paired per-face, which is the test that matters -- an aggregate mean can move
because a handful of faces changed a lot:

```
904 faces   B better on 747 (82.6%)   sign-test z = 19.6
            worse by >10% on 5 faces, better by >10% on 102
```

And with the confound removed. The new corpus CONTAINS the old one, so ~30% of
the held-out split is gen6's own training data, which flatters gen6. Shards
sort old-first, so those faces separate cleanly:

```
609 faces neither model trained on:  +4.52%,  B better on 81.9%,  z = 15.8
```

The advantage is slightly LARGER once gen6 stops being scored on its own
training data, which is the direction a real effect should move.

Val also stopped turning up -- gen7 ends flat at 0.02945/0.02947 rather than
bottoming and rising -- so the overfitting the corpus size was causing is gone.
Against a re-measured ceiling (the autoencoder, which sees the target, scores
0.0383 on this split) the generator now covers 84% of what this representation
can do, up from 82%. That ceiling is itself understated: `texgen_ae2` was
trained on the old corpus, so retraining it on 18,082 would likely lower it and
reopen some headroom.

**Gating the photometric loss on segmented skin changed nothing measurable.**
The photometric mask is the rendered mesh silhouette restricted to face-skin
triangles, which is mesh-side: it selects which triangles to compare, not which
photograph pixels are skin. Measured, that leaks 16.6% of loss pixels on FFHQ
and 21.6% on DigiFace to hair, clothes and background
(`scripts/eval/diag_photometric_mask.py`), so roughly a fifth of the
photometric signal was the encoder being asked to explain non-skin with a skin
albedo model, with geometry as the free variable available to do it.

Two matched FFHQ runs, 2 epochs, identical but for the mask:

| alpha | mask off | mask on | delta |
|---|---|---|---|
| 0.25 | 1.3244 | 1.3437 | +0.0194 |
| **0.40** | **1.3240** | **1.3405** | +0.0164 |
| 0.60 | 1.3605 | 1.3554 | -0.0051 |

No effect. Every delta sits inside the ~0.028 mm this setup can resolve, and
**the sign flips at alpha 0.60** -- a difference that changes direction across
the sweep is noise, not signal. Both arms peak at alpha 0.40 and both beat the
mean face (1.3693), so the pilots did learn; the mask did not change what.

The training loss is not the place to look for the answer here, and it is worth
saying why: the two arms compute the photometric term over DIFFERENT masks, so
the treatment's 13% lower photometric loss is arithmetic, not improvement. It
excluded exactly the pixels that are hardest to explain. Only an external metric
can settle it, which is what the sweep is for.

Replicated on the shipped configuration. The FFHQ pilot could not see the
mechanism it was testing, so the run was repeated with the identity corpus and
the full loss set -- 7,575 DigiFace subjects, K=4, shape swap 1.0, identity 0.2,
eye 1.0, lip 0.5, FFHQ mixed in, 3 epochs, masks on BOTH corpora:

| alpha | mask off | mask on | delta |
|---|---|---|---|
| 0.25 | 1.2472 | 1.2558 | +0.0086 |
| **0.40** | **1.2449** | **1.2484** | +0.0034 |
| 0.60 | 1.3009 | 1.2903 | -0.0106 |

Same verdict, and now where the mechanism was most plausible: every delta inside
the ~0.028 mm this setup resolves, the sign flipping again at 0.60, and
validation landmark loss identical to four decimals (0.0321 both). Two
independent pilots, on different corpora and different loss sets, both find
nothing. Gating the photometric loss on segmented skin does not improve
identity shape here.

A caveat on the absolute numbers, which is itself a finding: these two runs were
launched WITHOUT `FACE3D_MODEL`, so `assets.py` resolved FLAME implicitly and
they trained against FLAME 2020, not 2023 Open. The A/B is unaffected -- both
arms share the basis, and the mask is the only difference -- but 1.2449 must be
read against FLAME 2020's mean face at 1.3554, not against deca_open's 1.2798 on
a different basis. The model card warns that this argument is not optional and
it still caught us; these checkpoints are also not licence-clean and must not
ship.

What this pilot cannot rule out. It is FFHQ-only with no identity loss and no
shape swap, so it tests the photometric term in isolation rather than the
shipped configuration. The leak is larger on DigiFace (21.6%), which is where
the shape swap lives and therefore the mechanism most likely to convert cleaner
photometric signal into better identity shape. And NoW scores a NEUTRAL mesh, so
it is an identity-shape metric -- if the mask mainly cleans up expression or
albedo, the metric is blind to it by construction. The feature is kept behind
`--photo-mask`, off by default, with the leak measurement as the reason it might
still be worth revisiting on an identity-grouped corpus.

**A better identity signal is not the same as a better 3D shape.** MICA
replaces the pixel-derived shape head with an ArcFace identity embedding and
beats DECA while training on ~2,300 subjects. Probed on NoW first, the premise
held: between/within subject spread of 1.56 for the ArcFace embedding against
0.76 for our encoder's shape output.

Wired in as `ArcFaceShapeEncoder` -- shape from a cached 512-d embedding, the
other five output groups still from the ResNet trunk, everything else matched
to the pixel-based run:

| | pixels (digi_full) | ArcFace embedding |
|---|---|---|
| identity ratio on NoW | 0.76 | **0.85** |
| best NoW median | **1.3109 mm** | 1.3409 mm |

Best identity separation the project has produced, and a worse 3D score. The
embedding encodes what distinguishes faces for *recognition* -- much of it
texture and feature spacing that does not map onto FLAME's geometry basis.
Converting "who this is" into "what shape this is" is a separate learned
mapping, and that mapping is precisely what MICA's 3D-supervised training on
registered scans supplies. Learning it instead from landmark and photometric
losses, which are measurably near-blind to shape, does not work.

**Identity scale helps, but shallowly.** Best NoW median against the
FLAME mean face at 1.3554 mm, each at its own optimal shape scale:

| identities | best median | vs mean face | optimal alpha |
|---|---|---|---|
| 2,000 | 1.3300 mm | -1.9% | 0.25 |
| 10,000 | 1.3241 mm | -2.3% | 0.50 |
| **110,000 (all of DigiFace)** | **1.3109 mm** | **-3.3%** | 0.50 |

55x the identities bought 1.4%. The calibration optimum moved 0.25 -> 0.50
between the first two and then stopped, so the encoder is no better calibrated
at 110k than at 10k. Extrapolating this curve to SC1's 1.20 mm target would
need orders of magnitude more identity-grouped data than DigiFace contains.

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

`scripts/tools/visualize_predictions.py` shows the same thing directly: five subjects,
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
regularisation), not a different weight. `scripts/eval/diag_identity.py` reports the
between/within ratio precisely because the millimetre figure alone would have
called this a success.

**Landmarks and albedo are substitutes.** Direct parameter optimisation against
a textured synthetic target, 1200 iterations, only the fitted model varying
(`scripts/eval/ablation.py`):

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
