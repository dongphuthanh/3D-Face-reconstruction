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
| Rasteriser (batch 8, 224px) | 129 img/s, 937 MB VRAM |
| Training step (batch 8, 224px) | 163 ms, 2.12 GB peak |
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

**Three bugs found, all the same species: correct values, dead gradients.**

1. `torch.where(θ < ε, I, R)` in Rodrigues returns the right rotation at zero
   pose but a *constant*, so autograd sends zero gradient. Every training run
   initialises at zero pose — jaw, neck and head rotation would never have
   learned.
2. Fixing (1) naively produced NaN: `torch.where` evaluates the discarded branch
   anyway, and `inf × 0 = NaN`. Every denominator needs the masked value.
3. A zero-initialised regression head makes `∂L/∂trunk = ∂L/∂out · W` identically
   zero, starving the entire backbone.

None would have been caught by testing outputs. Hence the gradient-level tests.

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

**This project is non-commercial.** DECA's licence forbids distributing the
model, so shipping DECA-derived weights to a browser is not possible; on-device
inference is gated on training an encoder against FLAME 2023 Open.

Still needed before training on real photographs: the FLAME landmark embedding,
a 2D landmark detector, the BFM albedo basis, a face-parsing network, and an
identity-labelled image corpus.

## Evaluation

NoW data is complete locally (100 subjects, 20 validation scans with 7-point
landmarks). The official `now_evaluation` metric code pins numpy 1.19.5 and
chumpy 0.70 against Python ≤3.9 and needs a Unix C++ toolchain, so it runs
through the Dockerfile it ships rather than natively. Not yet wired up.

Use the **non-metrical** protocol: published DECA numbers are non-metrical, and
comparing against the metrical leaderboard produces a large unexplained gap.
