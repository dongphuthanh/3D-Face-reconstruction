# Architecture and decision record

What every module in `face3d/` does, how a photograph becomes a rigged head, and
the decisions behind it — with the measurements that justified each one.

Written to be studied. If you can explain §1, §3 and §5 from memory you can
defend this project in an interview.

---

## 1. The pipeline, end to end

### Inference: photograph → `.glb`

```
  JPEG bytes
     │
     ├─ learn/detect.py         MediaPipe → 478 2D landmarks, of which 105 are the
     │                          points the FLAME embedding was built on;
     │                          crop square at margin 1.6
     │                          Crop margin MUST match training (1.6) or it is a
     │                          distribution shift. This has bitten the project before.
     │
     ├─ learn/encoder.py        ResNet-50 → ONE linear head → 242 numbers
     │                          sliced into FlameParams (see §2)
     │                          × SHAPE_CALIBRATION = 0.40 on the shape block
     │
     ├─ geometry/flame_torch.py FLAME: shape+expr coefficients → 5023 vertices
     │                          blendshapes → joint regression → LBS → posed mesh
     │
     ├─ texture/texgen.py       ResNet-34 → bounded residual over the PCA albedo
     │  texture/skin.py         basis + 11 named skin parameters → 512² texture
     │  render/albedo.py        (the PCA basis itself; eyes come from render/eyes.py)
     │
     ├─ geometry/rig.py         identity-dependent joint positions + 20 expression
     │                          morph targets as deltas from the neutral mesh
     │                          (+ jaw_open added at export = 21 in the GLB)
     │
     └─ export/gltf.py          GLB: 5118 verts, 9976 tris, 2 primitives,
                                5 joints, 21 morph targets, embedded 512² PNG
```

**Note what is *not* in that list: the rasteriser.** `render/render.py`,
`render/pipeline.py` and `render/raster_cuda.py` execute zero lines during a
reconstruction. Export needs geometry and a texture, not a rendered image. The
rasteriser exists for training, and for the projection path below.

### Training: the loop that produced the checkpoint

```
  photo → encoder → FlameParams → FLAME → render/pipeline.py → rendered image
                                                   │
                                   compare against the photo, backprop
                                                   │
  learn/losses.py:  landmark (1.0) · photometric (2.0) · light reg (1.0)
                    identity (0.2) · eye-closure (1.0) · lip (0.5) · shape-swap (1.0)
```

FLAME is **frozen** — its arrays are registered as buffers, so
`model.parameters()` is empty and no optimiser can touch the basis by accident.
Gradients flow *through* it to the encoder.

### The texture path (two modes)

- **`texgen=True` (ships).** The generator predicts the texture from the photo
  directly. Fast, and on-manifold by construction.
- **`texgen=False` (projection).** `texture/project.py` runs the renderer
  *backwards*: rasterise in UV space, sample the photo through the mesh, gate
  occluders with `texture/segment.py`, de-light, mirror, frequency-merge with the
  basis. **This is how the training targets for the generator were made.**

---

## 2. `FlameParams` — the contract

The single most important type. Everything upstream and downstream agrees on it.

| field | shape | meaning |
|---|---|---|
| `shape` | (B, 100) | identity, PCA coefficients |
| `expr` | (B, 50) | expression, same convention |
| `pose` | (B, 15) | axis-angle × 5 joints: global, neck, jaw, L eye, R eye |
| `cam` | (B, 3) | weak perspective: scale, tx, ty |
| `light` | (B, 9, 3) | order-2 spherical harmonics, per RGB channel |
| `albedo` | (B, 50) | PCA albedo coefficients |
| `eye` | (B, 6) | iris RGB, then sclera RGB |

**The encoder emits 242 values**, not 236 and not 15 pose values:

```
100 shape + 50 expr + 6 rot + 3 cam + 27 light + 50 albedo + 6 eye = 242
```

Only **6** pose values are predicted — global rotation and jaw. They expand into
the 15-value vector with neck and both eyes left at rest. That is DECA's
convention, and it is why `params.jaw` is `pose[:, 6:9]`.

> **Interview trap 1:** "you said 5 joints but predict 6 numbers" — the rig has
> 5 joints, the encoder drives 2 of them (global rotation and jaw), and the
> other 3 are animated by the consumer through the exported armature.
>
> **Interview trap 2:** "FLAME has 300 shape components, why do you use 100?"
> The basis offers 300 shape and 100 expression; the encoder truncates to the
> first 100 and 50, which is what the published models do. PCA components are
> ordered by variance, so the tail contributes little — and every extra
> coefficient is another direction an under-constrained monocular objective can
> push in the wrong direction. Given D2 (the model is already over-confident at
> 100), widening it would make calibration worse, not better.

---

## 3. Module reference

### `geometry/` — FLAME and the rig

| module | lines | what it does |
|---|---|---|
| `flame_np.py` | 143 | NumPy FLAME. **Two roles:** it loads the MPI pickle (so it is a runtime dependency, not just a test double), and it is the oracle the torch port is diffed against — parity 8.3e-17 m. |
| `flame_torch.py` | 117 | Differentiable FLAME. Shape/expr blendshapes → joint regressor → `batch_rodrigues` → LBS, then cancels the rest-pose transform so zero pose is exactly identity. |
| `params.py` | 104 | `FlameParams` (§2). Validates batch consistency, pose width 15, light shape (9,3) at construction. |
| `rig.py` | 87 | FLAME → riggable asset: `rest_joints` (identity-dependent — a bigger head puts the neck joint elsewhere), `expression_targets` (20 morph deltas), `jaw_target`. |
| `landmarks.py` | 49 | The FLAME landmark embedding: barycentric coordinates mapping MediaPipe's 2D landmarks onto FLAME's surface. |
| `facemask.py` | 77 | Which FLAME vertices are face skin, and which are eyes. Used to restrict losses and texture operations to skin. |

### `render/` — differentiable image formation

| module | lines | what it does |
|---|---|---|
| `render.py` | 202 | The rasteriser: `weak_perspective`, `rasterize` (face ids + barycentrics), `interpolate`, `vertex_normals`, `sh_shading`. Pure PyTorch, mirrors nvdiffrast's interface. |
| `raster_cuda.py` | 113 | JIT-loads the CUDA kernel on first use, falls back to PyTorch with a *stated reason* if no toolchain. `FACE3D_NO_CUDA_RASTER=1` forces the fallback. |
| `pipeline.py` | 125 | Composes FLAME + camera + rasteriser into one differentiable photo-forming function. This is what the photometric loss backprops through. |
| `albedo.py` | 226 | The FLAME/BFM texture space as a differentiable albedo model. Also `face_texel_mask` (which UV texels the photometric loss optimised) and `harmonise`. |
| `eyes.py` | 134 | A parametric, differentiable eye texture — **not** an encoder. Exists because the 50-component albedo PCA allocates variance by *pixel area*, and the eyes are 15% of the UV map with almost no basis energy, so every face got the same blurred brown iris. |

### `texture/` — skin, the largest subsystem

| module | lines | what it does |
|---|---|---|
| `project.py` | 417 | **The biggest module and the central trick.** Photo → UV by running the renderer backwards. `uv_rasterize`, `mirror_uv_map` (see D11), `delight`, `project_photo`, `frequency_merge`, `composite`. |
| `texgen.py` | 268 | The learned generator: ResNet-34 trunk → bounded residual (`tanh × AMPLITUDE`) over the PCA basis + 11 parameters. Also the autoencoder used to establish the achievable ceiling, and `diffuse_fill` (pull-push pyramid). |
| `skin.py` | 198 | 11 named parameters → texture, differentiably: skin tone, freckles, crease depth, lip and brow colour. `compose`, `settle_unobserved`. |
| `regions.py` | 192 | Named face regions as soft UV masks; `unobserved_mask` (by surface *orientation*); `crease_map` derived from the corpus. |
| `segment.py` | 66 | MediaPipe portrait segmentation. Class ids `BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHER`. Gates occluders out of the projection. |

### `learn/` — the training side

| module | lines | what it does |
|---|---|---|
| `encoder.py` | 402 | `Encoder` protocol, `ResNetEncoder`, `ArcFaceShapeEncoder`. Holds `SHAPE_CALIBRATION`. Also a `load_state_dict` that pads old 236-wide checkpoints to 242 rather than orphaning them — and refuses to truncate. |
| `losses.py` | 165 | landmark, photometric, `IdentityLoss` (facenet InceptionResnetV1), closure (eye/lip), shape-consistency, regularisation. |
| `augment.py` | 211 | Paired-view augmentation, `two_views`, `swap_shape` — the DECA-style shape swap. |
| `data.py` | 214 | `FFHQCrops`, `IdentityPairs` — datasets over the ingested crops. |
| `detect.py` | 116 | MediaPipe landmarks, `crop_square`. **Not thread-safe** — MediaPipe graphs hold state, so the webapp keeps one per thread. |

### `export/` and root

| module | lines | what it does |
|---|---|---|
| `gltf.py` | 336 | glTF 2.0 / GLB writer: skinning, morph targets, embedded PNG, seam-split vertices. Validator-clean. |
| `now.py` | 130 | NoW benchmark harness: prediction layout, the 7 alignment landmarks, Docker invocation. |
| `assets.py` | 62 | Locates a FLAME model. **Pass the basis explicitly** — implicit resolution is how the project shipped FLAME 2020 geometry while believing it was 2023 Open. |

---

## 4. Decisions you will be asked about

### D1. Why FLAME, and why is it frozen?

A 3D morphable model is a strong prior: it guarantees the output is a plausible
head no matter what the encoder predicts. Monocular depth is underdetermined;
the basis is what makes the problem tractable at all. Freezing it means the
model can only move *within* the space of real faces, and it makes the export
rig well-defined.

### D2. `SHAPE_CALIBRATION = 0.40` — the most important line in the repo

The raw prediction scores **worse than emitting a constant mesh**. Keeping only
40% of the predicted shape deviation is the only reason the model beats the mean
face. The encoder is over-confident; the constant is the measurement that says
so. Applied in `predict(calibrate=True)`, off during training so the loss sees
the raw prediction.

**Be honest about this in an interview.** It is a calibration constant covering
for a weak identity signal, and saying so is much stronger than pretending it is
a tuning nicety.

### D3. Landmarks are a pose signal, not a shape signal

Swap one parameter group for another sample's value, measure how far the 105
projected landmarks move:

| swapped | pose | camera | shape | expression |
|---|---|---|---|---|
| landmark shift | **17.13 px** | 3.70 px | **1.95 px** | 1.41 px |

Pose moves landmarks ~9× more than shape does. So a landmark-dominated
objective is nearly blind to identity. The original weighting was landmark
5 : photometric 1; DECA's is 1 : 2 — a 10× swing toward the term that actually
sees shape. Adopting DECA's config moved the identity ratio 0.76 → 0.96, the
largest single gain in the project, and it cost nothing but reading their code.

### D4. The consistency loss that improved the metric and broke the model

A shape-consistency term across augmented views improved NoW by 15% and made
identity learning *worse*: within-subject shape spread fell 3×, between-subject
spread fell **4.4×**. `(shape_A − shape_B)²` is minimised perfectly by
predicting a constant, and nothing in the objective rewarded shape varying
between people. The NoW "gain" was regression toward the mean face, which
outscores both runs.

**This is the best story in the project.** It is why every claim is now checked
against (a) a baseline needing no model — the mean face — and (b) a second
metric measuring the intended effect (between/within identity ratio).

### D5. Landmarks and albedo are substitutes

Direct parameter fitting against a synthetic target, 1200 iterations:

| | no landmarks | real landmarks |
|---|---|---|
| grey albedo | 30.14 mm | 5.45 mm |
| texture albedo | 6.18 mm | 5.43 mm |

Either signal alone rescues the fit; both together add almost nothing. Albedo's
value is as an *independent* path — which matters when landmarks fail on
profiles and occlusions.

### D6. Why the texture is generated, not projected

Raw projection is faithful, which is the problem: it transfers whatever the
photograph contains onto a head that cannot represent it. Auditing 14 FFHQ
faces, **8 carried an occluder** — spectacle frames, a fringe, a beaded
headdress, hair over a cheek. Occluders, not smearing, were the dominant
artefact.

So projection became the *training target*, and the shipped path is a generator
predicting a **bounded** residual (`tanh × amplitude`) over the PCA basis plus 11
named parameters. The bound is the guarantee: the output cannot leave the
manifold of plausible skin. Similarity is deliberately traded for that.

| | masked L1 |
|---|---|
| PCA basis alone | 0.1497 |
| **generator + procedural layer** | **0.0594** — 60% closer |
| autoencoder that *sees* the target | 0.0403 — the achievable ceiling |

### D7. Eyes get their own parameterisation

The 50-component albedo PCA allocates variance by pixel area, so eyes (15% of
the UV map) carry almost no basis energy and every face got a population-average
brown blob with no sclera. Iris colour is one of the strongest identity cues
available, and it was being discarded.

Iris RGB is free through a sigmoid. **Sclera is clamped** to
`[0.88, 0.85, 0.82] ± 0.09` — and that is not timidity. Left free it collapsed
to `[0.47, 0.34, 0.29]`, a mid-brown: at 20×10 px the eye region is mostly
eyelid and lash, so a darker eyeball genuinely lowers pixel error while ceasing
to look like an eye. Anatomy is the correct prior; nobody has brown whites.

Result: iris colour correlates with the photographed iris at **r = +0.822** over
80 faces.

### D8. The CUDA rasteriser

One thread per triangle, bounding-box traversal, and a lock-free depth test
resolving winner *and* tie-break in a single `atomicMin` on a packed 64-bit
`(depth, face_id)` key. **52× faster** than the tensor-op version, **25% off the
training step**, and **bit-exact** — 0 mismatched pixels over 8 fixture cases and
4 configurations.

Bit-exactness is a *consequence* of the packed key: the face index sits in the
low bits, so ties break deterministically to the lower index.

Nsight Compute later showed FP64 as the top pipeline in a kernel with no doubles
by design — traced to `y + 0.5`, where a bare `0.5` is a double literal that
promotes the whole expression. Consumer Blackwell runs FP64 at 1/64 rate. `0.5f`
**doubled throughput**; verified with `cuobjdump -sass`.

DECA ships the same architecture with one difference: theirs does `atomicMin`
then a *separate* re-read to decide whether to write the face index, which
races — depth buffer and index buffer can disagree, and exact ties let two
threads both write.

> **Why was PyTorch slow?** Rasterisation is ragged — each triangle covers a
> different number of pixels, writes are scattered atomics. Dense tensor ops
> cannot express that, so every triangle is padded to a `k×k` block: 7.46M
> candidate tests vs 1.73M for true bounding boxes, 4.3× more arithmetic before
> any memory traffic, plus a DRAM round-trip per intermediate where the kernel
> uses 40 registers with zero spill.
>
> **Never say "52× faster than PyTorch."** PyTorch ships no rasteriser; the
> baseline is a hand-written tensor-op implementation in this repo.

### D9. FLAME 2020 vs 2023 Open — why retraining was unavoidable

The two bases share topology (5023 verts, 9976 faces, byte-identical `faces`)
and are equally expressive. But **their axes are rotated relative to each
other**: expressing either basis's top-100 directions in the other's span
retains only **81%** of the energy, in both directions. Coefficients mean
different things, so a 2020-trained encoder emits nonsense through a 2023 basis.

Accuracy was not the cost — 2023 Open is not the weaker basis. The cost was one
retraining run, and the gain is a licence-clean model whose meshes may be
redistributed.

### D10. Input quality dominates everything

Three photographs of one person:

| input | recognisability |
|---|---|
| studio portrait, no glasses, 700px | **0.715** |
| outdoor, harsh side light, 400px | 0.529 |
| glasses, banner overlay, 400px | 0.396 |

**The spread between a good and a bad photograph of the same person is larger
than the spread between any two models trained in this project.** A capture
guide is worth more than further training.

Averaging shape across multiple photographs does *not* help (0.525 → 0.477):
per-image error is systematic bias, not independent noise, so averaging mixes
biases in rather than cancelling them.

### D11. The FLAME template is not mirror-symmetric

Half a face is often occluded, so the projection mirrors what it saw across the
midline. The obvious implementation pairs vertices combinatorially — assume
vertex *i* has a mirror twin found by negating x.

**That assumption is false.** Measured on the template: median pairing error
0.00115 against a mean edge length of 0.00331, and **775 vertices with no
partner at all**. FLAME's mesh is close to symmetric, not symmetric.

The fix is geometric, not combinatorial: a scipy `cKDTree` over the x-negated
vertex positions, with a normal-agreement test to reject false matches where two
surfaces come close (eyelid against eyeball). Built once and cached, since it
depends only on topology.

**Why it matters as an answer:** it is a case of an assumption that is *almost*
true, where "almost" produces a subtly wrong texture rather than a crash. The
only way to find it is to measure the assumption instead of trusting it.

### D12. Why MICA beats everyone with less data

MICA scores 0.98 mm from **2,315 subjects with real 3D scan supervision**,
against DECA's ~2M images with none. Three orders of magnitude less data, a
better result. That is the strongest available evidence that the bottleneck is
the *kind* of supervision, not the amount — and it is consistent with six corpus
experiments here all landing inside 1.24–1.29 mm.

It is also why the ArcFace experiment failed the way it did: swapping in an
identity embedding gave the best identity separation the project produced (0.85)
and a *worse* NoW score. Face-recognition embeddings encode what distinguishes
faces for recognition — much of it texture and feature spacing that does not map
onto FLAME's geometry basis. Converting "who this is" into "what shape this is"
is a separate learned mapping, and that mapping is exactly what 3D scans supply.

---

## 5. Numbers to know cold

| | |
|---|---|
| FLAME 2023 Open | 5023 verts, 9976 faces, 5 joints |
| basis components *available* | 300 shape, 100 expression |
| components the encoder *uses* | first **100** shape, **50** expression |
| encoder output | 242 = 100 + 50 + 6 + 3 + 27 + 50 + 6 |
| NoW, mean face (no model) | **1.3693 mm** |
| NoW, this project | **1.2798 mm** |
| NoW, DECA / MICA | 1.09 / 0.98 mm |
| available headroom | ~0.39 mm; this takes ~23% of it |
| texture, basis → ours → ceiling | 0.1497 → **0.0594** → 0.0403 |
| CUDA rasteriser | 52×, 25% off the step, bit-exact |
| exported GLB | 5118 verts (5023 + 95 seam-split), 9976 tris, 2 primitives, 21 morph targets |
| identity ratio (between/within) | 1.14 |
| crop margin | 1.6, everywhere, non-negotiable |

---

## 6. Four bugs that produced plausible output

All the same species: renders looked fine, internals were broken. This is the
answer to "tell me about a hard bug".

1. **`torch.where(θ < ε, I, R)` in Rodrigues** returns the correct rotation at
   zero pose but a *constant* — so autograd sends zero gradient. Every run
   initialises at zero pose, so jaw, neck and head rotation would never have
   learned. The renders looked right the whole time.
2. **Fixing (1) naively produced NaN.** `torch.where` evaluates the discarded
   branch anyway, and `inf × 0 = NaN`. Every denominator needs the masked value.
3. **A zero-initialised regression head** makes `∂L/∂trunk = ∂L/∂out · W`
   identically zero, starving the entire backbone. Fixed with `std=3e-4` — small
   enough to still start at the mean face, non-zero enough to train from step one.
4. **FLAME's texture space stores channels BGR.** Faces rendered blue —
   structurally perfect, wrong colour. The photometric loss would have absorbed
   it into distorted albedo coefficients rather than failing.

Hence tests that assert on **invariants** — gradients non-zero at
initialisation, skin satisfying R > G > B — not on whether a render looks like a
face.

---

## 7. Weak points — know these before someone finds them

Being able to state these unprompted is worth more than the strengths.

- **Identity shape is weak.** Over 200 faces the output sits 2.18 mm from the
  mean face while two *different* people's outputs differ by only 1.79 mm —
  signal is ~1.4× noise. Rendered neutral, a toddler and an elderly woman come
  out visibly similar. `SHAPE_CALIBRATION = 0.40` is the admission of this.
- **The model does not beat DECA**, and is ~0.3 mm behind it. The defensible
  claim is the evaluation discipline, not the score.
- **Seed variance was never measured.** Any gap below ~0.03 mm in this project
  should be treated as a tie.
- **The CUDA kernel does nothing at inference.** It makes training 25% faster.
  Say this before you are asked.
- **`make` has never run on the development machine** — the Makefile is
  validated by CI and by running the recipe bodies directly.
- **Cannot represent**: eyewear, facial hair, hair, tongue, ears in detail.
  FLAME has no basis for them. A scalp shell was built, measured, judged worse
  than bald, and deleted.
- **Privacy**: since the texture became a projection, an exported GLB contains
  the photograph itself wrapped onto a mesh. That is a different object from 50
  PCA coefficients — the earlier bake could not reproduce a recognisable image
  of anybody, and this one is one by construction.

---

## 8. Likely questions, short answers

**"Walk me through what happens when I upload a photo."** → §1.

**"Why is there a NumPy FLAME *and* a torch FLAME?"** → `flame_np` loads the
pickle and is the oracle the torch port is diffed against — 8.3e-17 m parity.
Reimplementing LBS is exactly the kind of code that is silently wrong.

**"How do you know your rasteriser is correct?"** → Bit-exact against the
reference: 0 mismatched pixels over 8 fixtures (degenerate triangles, off-screen
clipping, sub-pixel geometry, exact depth ties) and 4 configurations. Achievable
*because* the packed key makes tie-breaks deterministic.

**"What's the hardest bug you fixed?"** → §6, item 1 or the FP64 find in D8. The
FP64 one is the better story: hypothesis → profiler → root cause → one-character
fix → independent verification in the disassembly.

**"How do you know your model is any good?"** → Against a baseline that needs no
model. Predicting the average face scores 1.3693 mm; this scores 1.2798. And a
second metric, because §D4 happened.

**"What would you do differently?"** → Get 3D scan supervision. Six corpus
experiments all landed in 1.24–1.29 mm; MICA reaches 0.98 with 2,315 scanned
subjects. The bottleneck is the kind of supervision, and no amount of
self-supervised photographs fixes it.

**"What's the weakest part?"** → §7, and lead with identity shape.
