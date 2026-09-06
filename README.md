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

**How close is the mesh?** NoW benchmark, non-metrical protocol:

| | median error |
|---|---|
| FLAME mean face — ignores the photo entirely | 1.3693 mm |
| **this project** | **1.2798 mm** |
| DECA | ~1.09 mm |
| MICA — best published | 0.98 mm |

The baseline matters more than the score. Predicting the *average face* gets you
1.3693 mm, so the entire band worth competing in is ~0.39 mm wide and this takes
about a quarter of it. Monocular identity shape is underdetermined; every
published result sits close to that constant baseline. MICA beats everyone with
three orders of magnitude *less* data — 2,315 subjects with real 3D scans —
which says the bottleneck is the kind of supervision, not the amount.

**How good is the texture?** Mean absolute error against the photograph, over
the face region:

| | |
|---|---|
| the FLAME albedo basis alone | 0.1508 |
| **this project** | **0.0564** — 63% closer |

Measured on 904 held-out faces. For scale, an autoencoder allowed to *see* the
target reaches 0.0383, so this covers about **84%** of what is achievable with
this representation.

One number is load-bearing enough to name: `SHAPE_CALIBRATION = 0.40`. The raw
prediction scores **worse than a constant mesh**; keeping only 40% of the
predicted shape deviation is what makes the model beat the mean face. The
encoder is over-confident, and that constant is the measurement that says so.

See [MODEL_CARD.md](MODEL_CARD.md) for the shipped checkpoint and its limits,
[FINDINGS.md](FINDINGS.md) for how each gain was won or lost.

## Engineering highlights

### Custom CUDA rasteriser

The differentiable rasteriser was pure PyTorch (nvdiffrast does not build on
this machine), and its z-buffer pass dominated training. Rewritten as a CUDA
kernel: **one thread per triangle, bounding-box traversal, and a lock-free depth
test that resolves the winner and the tie-break in a single `atomicMin` on a
packed 64-bit `(depth, face_id)` key.**

**52× faster** than the equivalent tensor-op version, taking **25% off the
training step** — and **bit-exact** against it, zero mismatched pixels across
eight fixture cases and four batch/resolution configurations.

Bit-exactness is a *consequence* of the packed key, not a lucky result. The face
index sits in the low bits, so ties break deterministically to the lower index
and the two implementations agree pixel-for-pixel rather than approximately.

Profiling with Nsight Compute then found FP64 as the top pipeline in a kernel
that should contain no double-precision work at all — traced to `y + 0.5`, where
a bare `0.5` is a double literal that promotes the whole expression. Consumer
Blackwell runs FP64 at 1/64 rate. The fix was `0.5f`, verified in the
disassembly, and it **doubled kernel throughput**.

Architecturally this matches the rasteriser DECA ships, with one difference:
DECA's does `atomicMin` and then a *separate* re-read to decide whether to write
the face index, which races — the depth buffer and the index buffer can
disagree, and exact ties let two threads both write. Packing removes the window.
[cuda/RASTERISATION.md](cuda/RASTERISATION.md) has the maths and the fixtures.

### Texture: projection, then re-synthesis

Projecting the photograph into UV space directly gives high fidelity and ugly
failures — occluders, seams, and smeared texels wherever the surface turns away
from camera. Auditing 14 faces, 8 carried an occluder; that, not smearing, was
the dominant artefact.

So the projection is a *target*, not the output. A generator predicts a bounded
residual over a PCA basis (`tanh × amplitude`) plus 11 interpretable parameters
— skin tone, freckles, crease depth, lip and brow colour. The bound is the
point: whatever the photograph contains, the result cannot leave the manifold of
plausible skin. Similarity is traded for the guarantee that nothing weird
appears.

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

## Running it

```bash
make setup       # pinned deps, from the CUDA index matching the local toolkit
make check-cuda  # which torch build is installed, and does the kernel load
make test        # full suite against whichever FLAME model is on disk
make test-ci     # same suite against a synthetic fixture, no licensed assets

python -m uvicorn webapp.server:app --port 8000    # browser demo + /docs
bash cuda/test_raster.sh                           # rasteriser fixtures, 8 cases
```

The CUDA rasteriser is optional and self-installing: `face3d/render/raster_cuda.py`
JIT-compiles it on first use and falls back to the PyTorch path with a stated
reason if no toolchain is present. `FACE3D_NO_CUDA_RASTER=1` forces the
fallback. It requires a torch build matching the installed CUDA major version.

`make setup` installs torch from the CUDA index matching the local toolkit
(cu130 for CUDA 13.3) and then runs `make check-cuda`, which reports the build
and whether the kernel loads:

```
torch 2.9.0+cu130   cuda 13.0   gpu True
cuda rasteriser: ok
```

That check exists because the failure is quiet. A torch built against a
different CUDA does not error — the extension just falls back to the PyTorch
path and training runs ~25% slower with nothing in the log. If you are swapping
CUDA builds, use `make setup-force`: pip treats `2.9.0+cu128` and `2.9.0+cu130`
as the same version and will not replace one with the other.

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
                     segment       MediaPipe portrait segmentation, to gate occluders
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
  data/            download, detect, crop and cache the image corpora
  train/           train.py, train_overfit.py, train_texgen.py
  eval/            NoW prediction + harness validation, closure, ablations, diagnostics
  tools/           export a head, render a comparison, visualise predictions

webapp/            FastAPI service + Three.js viewer
```

---

## Assets — not included

Every asset is licensed and obtained separately, and gets a row in the licence
register before it enters the pipeline. The shipped model was built from these
and nothing else:

| Asset | Source | Trained models redistributable |
|---|---|---|
| FLAME 2023 Open | flame.is.tue.mpg.de | yes, with attribution (CC-BY-4.0) |
| DigiFace-1M | Microsoft | yes, explicitly (R-UDA v1.0) |
| FFHQ (permissive subset) | NVlabs | yes (CC BY / PD / CC0) |
| MediaPipe landmarker / segmenter | Google | yes (Apache-2.0) |
| NoW benchmark | now.is.tue.mpg.de | evaluation only |

That list is short on purpose. A clean basis plus a corpus that explicitly
permits redistributing trained models is what makes the exported meshes
shareable. No DECA-derived weights are used anywhere — where the code says "DECA
weights" it means their published *loss weights*, read from their config.

Earlier experiments in this project did use other corpora, including scraped-
provenance ones. Those runs are retired and ship nothing;
[FINDINGS.md](FINDINGS.md) records what they measured and
[MODEL_CARD.md](MODEL_CARD.md) why they were dropped.

**This project is non-commercial.** Face images are biometric data under GDPR
and BIPA: no scraping, consent on file for every demo photograph, and uploads
are never retained. Note that since the texture became a projection, an exported
GLB contains the photograph itself wrapped onto a mesh — sharing the file is
sharing the photograph.

CI runs against a synthetic model generated from an icosphere
(`tests/fixtures/make_fixture.py`) that shares FLAME's pickle structure and
contains no MPI data, so the full pipeline — LBS, joints, blendshapes,
rasterisation, gradients — is exercised without any licensed asset. Suites whose
thresholds are stated in millimetres against a real head skip themselves.

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
