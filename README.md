# face3d

**One photograph in, a rigged glTF head out.** Monocular 3D face reconstruction —
a ResNet-50 encoder regresses FLAME parameters, a hand-written differentiable
rasteriser closes the loop, and the result exports as a game-ready asset with a
skeleton, blendshapes and a baked texture.

Runs end to end on a single consumer GPU. Every number in this repository was
measured on that machine; none are quoted from papers.

```
                    ┌──────────── trainable ────────────┐
    photograph  →   ResNet-50   →   FlameParams   →   FLAME   →   rasteriser   →   losses
                                    shape, expr,      5023 v      CUDA z-buffer     landmark
                                    pose, cam,        9976 f      + torch bary      photometric
                                    light, albedo                                   identity
                                                          ↓
                                    glTF: 5 joints, 20 blendshapes, baked 512² texture
```

---

## What it produces

A `.glb` you can drop into Blender, Three.js or Unity: skinned, posed, with
working expression sliders and a texture derived from the input photograph.
A browser demo (`webapp/`) does photo-upload → live 3D preview with expression
and skin-tone controls.

| | |
|---|---|
| Geometry | FLAME 2023 Open — 5,023 vertices, 9,976 faces, 5 joints |
| Rig | jaw / neck / eyes, LBS weights carried through from the basis |
| Blendshapes | 20 expression morph targets |
| Texture | 512² baked, projected from the photo then re-synthesised (below) |
| Licence of shipped weights | clean — CC-BY-4.0 basis, no DECA derivatives |

---

## Results

**NoW benchmark, non-metrical protocol**, 20 validation subjects:

| | median | vs. its own baseline |
|---|---|---|
| MICA (3D-supervised, published) | 0.98 mm | — |
| DECA (published) | ~1.09 mm | — |
| **this project — `deca_open`** | **1.2798 mm** | **−0.0895** |
| FLAME mean face (2023 Open basis) | 1.3693 mm | baseline |

The honest framing: predicting the *average face* scores 1.3693 mm, and the
state of the art is 0.98 mm. The entire available headroom is ~0.39 mm, and this
model takes about a quarter of it. That gap is a data problem, not a tuning one
— MICA closes it with 3D scan supervision, which this project has no licence to
use. See [MODEL_CARD.md](MODEL_CARD.md) for checkpoint selection and
[FINDINGS.md](FINDINGS.md) for how each gain was won or lost.

**The single most important number in the repo:**

```python
SHAPE_CALIBRATION = 0.40      # face3d/learn/encoder.py
```

The raw prediction scores **1.5485 mm — worse than emitting a constant mesh**.
Discarding 60% of the predicted shape deviation is the only reason this beats
the mean face. The model is over-confident, and the calibration constant is the
measurement that says so.

---

## Engineering highlights

### Custom CUDA rasteriser

The differentiable rasteriser was pure PyTorch (nvdiffrast does not build on
this machine). Its z-buffer pass dominated training, so it was rewritten as a
CUDA kernel: **one thread per triangle, bounding-box traversal, and a lock-free
depth test that resolves the winner and the tie-break in a single `atomicMin`
on a packed 64-bit `(depth, face_id)` key.**

| | |
|---|---|
| vs. the equivalent tensor-op implementation | **51.8×** |
| encoder training step | **24.6% faster** (1.33×) |
| correctness | bit-exact — 0 mismatched pixels, 8 fixture cases + 4 batch/resolution configs |

Bit-exactness is a *consequence* of the packed-key design: because the face
index occupies the low bits, ties break deterministically to the lower index,
so the kernel and the reference agree pixel-for-pixel rather than approximately.

Profiling with Nsight Compute then found FP64 as the top pipeline at 63%
utilisation in a kernel that should contain no double-precision work — 271,511
FP64 instructions traced to `y + 0.5`, where the bare `0.5` is a double literal
promoting the whole expression. Consumer Blackwell runs FP64 at 1/64 rate.
The fix was `0.5f`; `cuobjdump -sass` confirms zero `DADD`/`DMUL`/`DFMA` remain.
That one character **doubled kernel throughput**.

Architecturally this matches the rasteriser DECA ships, with one difference:
DECA's does `atomicMin` and then a *separate* re-read to decide whether to write
the face index, which races — the depth buffer and the index buffer can
disagree, and exact ties let two threads both write. Packing removes the window.
[cuda/RASTERISATION.md](cuda/RASTERISATION.md) has the maths and the fixtures.

### Texture: projection, then re-synthesis

Projecting the photograph into UV space directly gives high fidelity and ugly
failures — occluders, seams, and smeared texels wherever the surface turns away
from camera. Auditing 14 faces, 8 carried an occluder; that, not smearing, was
the dominant artifact.

So the projection is a *target*, not the output. A generator predicts a bounded
residual over a PCA basis (`tanh × AMPLITUDE`) plus **11 interpretable
parameters** — skin tone, freckles, crease depth, lip and brow colour — which
guarantees the result stays on the manifold of plausible skin no matter what the
photograph contains:

| | masked L1 |
|---|---|
| PCA basis alone | 0.1497 |
| **generator + procedural layer (ships)** | **0.0594** — 60.3% closer |
| autoencoder that *sees* the target (upper bound) | 0.0403 — 73.1% closer |

Deliberately trading similarity for the guarantee that nothing weird appears.

### Evaluation that can say "no"

The project's recurring failure mode was a metric improving while the thing it
was supposed to measure got worse. A shape-consistency loss improved NoW by 15%
by making the encoder invariant to identity — it cut within-subject shape spread
3×, and between-subject spread 4.4×. `(shape_A − shape_B)²` is minimised
perfectly by predicting a constant.

So every claim is checked against a baseline that needs no model (the mean face)
and a second metric that measures the intended effect (between/within identity
ratio). [FINDINGS.md](FINDINGS.md) records where that changed a conclusion,
including the ones that went the wrong way.

### Bugs that produce plausible output

Four found, all the same species — the renders looked fine and the internals
were broken:

1. `torch.where(θ < ε, I, R)` in Rodrigues returns the correct rotation at zero
   pose but a *constant*, so autograd sends zero gradient. Every run initialises
   at zero pose; jaw, neck and head rotation would never have learned.
2. Fixing (1) naively produced NaN — `torch.where` evaluates the discarded
   branch anyway, and `inf × 0 = NaN`.
3. A zero-initialised regression head makes `∂L/∂trunk` identically zero,
   starving the backbone.
4. FLAME's texture space stores channels BGR. Faces rendered blue — structurally
   perfect, wrong colour. The photometric loss would have absorbed it into
   distorted albedo coefficients rather than failing.

None would have been caught by looking at outputs. Hence tests that assert on
invariants — gradients non-zero at initialisation, skin satisfying R > G > B —
rather than on whether a render looks like a face.

---

## Performance

RTX 5070 Laptop (Blackwell sm_120, 8 GB), torch 2.9.0+cu130, CUDA 13.3.

| | |
|---|---|
| FLAME forward (batch 32) | 16,000 meshes/s, 66 MB VRAM |
| Rasteriser z-buffer, CUDA (batch 8, 224px) | 0.177 ms |
| — same, tensor-op implementation | 9.17 ms |
| Encoder training step (batch 8, 224px) | 27.8 ms |
| Torch↔NumPy FLAME parity | 8.3e-17 m |

Timings are medians of 5 separate processes, 50 reps after 20 warmup.

---

## Running it

```bash
make setup     # pinned deps
make test      # full suite against whichever FLAME model is on disk
make test-ci   # same suite against a synthetic fixture, no licensed assets

python -m uvicorn webapp.server:app --port 8000    # browser demo + /docs
bash cuda/test_raster.sh                           # rasteriser fixtures, 8 cases
```

The CUDA rasteriser is optional and self-installing: `face3d/render/raster_cuda.py`
JIT-compiles it on first use and falls back to the PyTorch path with a stated
reason if no toolchain is present. `FACE3D_NO_CUDA_RASTER=1` forces the
fallback. It requires a torch build matching the installed CUDA major version.

> **Known gap:** `make setup` still pins the cu128 index while the CUDA
> extension needs cu130 to match CUDA 13.3. A fresh install gets the fallback
> path, not the kernel.

Blackwell needs a matching CUDA wheel either way — older ones install cleanly
and then fail at the first kernel launch.

---

## Layout

Six top-level pieces. Each subpackage is one sentence.

```
face3d/            the library — everything importable
  geometry/        FLAME: the statistical head model, its parameters, and the rig
                     flame_np      NumPy FLAME — the oracle the torch port is diffed against
                     flame_torch   differentiable FLAME, batched, CUDA
                     params        FlameParams — the typed encoder↔pipeline contract
                     rig           FLAME → armature + named morph targets
                     landmarks     the FLAME landmark embedding
                     facemask      which vertices are face skin
  render/          differentiable rendering — image formation and its gradients
                     render        rasteriser + SH shading
                     raster_cuda   JIT loader for the CUDA kernel, graceful fallback
                     pipeline      params → mesh → landmarks → shaded image
                     albedo        FLAME texture space as a differentiable albedo
                     eyes          parametric differentiable eye texture
  texture/         skin: photograph → UV, learned re-synthesis, procedural control
                     project       photo → UV texture, by running the renderer backwards
                     texgen        learned texture generator + autoencoder
                     skin          11-parameter procedural skin layer
                     regions       named face regions as soft UV masks
                     hair          scalp shell fitted to the photo (off by default)
  learn/           the training side
                     encoder       Encoder protocol + ResNet-50 regressor
                     losses        landmark, photometric, identity, consistency
                     augment       paired-view augmentation
                     data          datasets over the ingested crops
                     detect        2D landmarks from MediaPipe
  export/          getting results out
                     gltf          glTF 2.0 / GLB writer
                     now           NoW benchmark layout
  io/              gdrive, remotezip — fetch remote assets without whole archives
  assets.py        locates a FLAME model; lets tests skip when it is absent

cuda/              the hand-written rasteriser
  raster_kernel.cuh  the kernel, shared by harness and extension
  raster_ext.cu      PyTorch binding
  raster.cu          standalone harness + per-phase benchmark
  test_raster.sh     8 fixture cases, PASS/FAIL
  profile.sh         Nsight Compute harness
  RASTERISATION.md   the maths, and pseudocode for the algorithm

tests/             every check — run_tests.py runs them in dependency order
  fixtures/          the synthetic FLAME model, and the rasteriser fixtures

scripts/           entry points, grouped by what they are for
  data/            ingest FFHQ / CelebA / DigiFace / Arc2Face, build corpora
  train/           train.py, train_overfit.py, train_texgen.py
  eval/            NoW prediction + harness validation, closure, ablations, diagnostics
  tools/           export a head, render a comparison, visualise predictions

webapp/            FastAPI service + Three.js viewer
```

---

## Assets — not included

Every asset is licensed and must be obtained separately; every one has a row in
the register in §9 of the implementation plan before it enters the pipeline.

| Asset | Source | Redistributable |
|---|---|---|
| FLAME 2020 / 2023 | flame.is.tue.mpg.de | No |
| FLAME 2023 Open | same | Yes (CC-BY-4.0) |
| NoW benchmark | now.is.tue.mpg.de | No |
| DECA weights | DECA repo | **No** |
| FFHQ (permissive subset) | NVlabs | Yes (CC BY / PD / CC0) |
| DigiFace-1M | Microsoft | Data no; trained models yes (R-UDA) |
| Arc2Face | HuggingFace | CC BY-NC-SA 4.0, ShareAlike |
| CelebA | MMLAB CUHK, via `flwrlabs/celeba` | **No** — non-commercial, no redistribution |

**This project is non-commercial.** Arc2Face and CelebA both derive from scraped
source images; the project's no-scraping rule is traded knowingly in both cases
and recorded here rather than glossed. CelebA additionally forbids
redistribution outright and gates its identity annotations behind a request the
public mirror skips, so only the recipe is tracked, never the data.

CI runs against a synthetic model generated from an icosphere
(`tests/fixtures/make_fixture.py`) that shares FLAME's pickle structure and contains no
MPI data, so the full pipeline — LBS, joints, blendshapes, rasterisation,
gradients — is exercised without any licensed asset. Suites whose thresholds are
stated in millimetres against a real head skip themselves.

---

## How this was built

Written with substantial help from [Claude](https://claude.com/claude-code)
(Anthropic), used as a pair-programming and research assistant: drafting code,
running the training and evaluation sweeps, and writing up results. The
direction, the questions worth asking, and the judgement about which results to
trust were mine.

Where that shows in the repo, deliberately: commit messages and code comments
record *why* a thing is the way it is, including the experiments that failed.
Several conclusions were wrong on the first measurement and were corrected —
they are left in rather than tidied away.
