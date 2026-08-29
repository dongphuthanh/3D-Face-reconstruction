# Monocular 3D Face Reconstruction — Implementation Plan

**Status:** Draft · **Owner:** _(you)_ · **Last updated:** 2026-08-20
**Timeline:** 12 weeks @ ~10–15 hrs/week · **Hardware:** 1× consumer GPU or Colab

---

## 1. Problem & Motivation

Existing single-image 3D face reconstruction research (DECA, SMIRK, MICA) produces
good meshes but stops at a research artifact: a `.obj` dumped to disk, a Python
script, and no path to a usable asset. There is a real gap between "a model that
predicts FLAME parameters" and "a rigged head I can drop into Blender or a game
engine."

This project closes that gap. The deliverable is a **library plus two frontends**
that take a photo and produce a rigged, blendshape-driven, engine-loadable head
asset — with measured accuracy, measured latency, and tests proving the export is
correct.

### Why this shape of project

The ML component is deliberately bounded. Training a face model from scratch is
GPU-weeks of work that a reviewer cannot evaluate. The engineering around it —
abstraction boundaries, export correctness, real-time constraints, failure
handling — is both tractable in the timeline and the thing actually being
assessed.

---

## 2. Goals

**G1.** Single image → rigged glTF head asset that loads correctly in Blender,
three.js, and Unity.

**G2.** Live browser demo, on-device inference, no image ever leaves the client.
_Amended 2026-08-20: on-device inference requires shipping weights to visitors,
which DECA's license forbids. Gated on training our own encoder — see §9._

**G3.** Real-time webcam tracking at a stated, measured latency budget.

**G4.** Reported reconstruction accuracy on a public benchmark (NoW), comparable
to published baselines.

**G5.** One core library consumed by two independent frontends, demonstrating a
real abstraction boundary.

## 3. Non-Goals

- Photorealistic rendering or relighting
- Full-head assets: hair, teeth interior, eyelashes, accessories
- Video-rate temporal smoothing beyond simple filtering
- Beating state of the art on any benchmark
- User accounts, persistence, multi-tenancy, or any backend service
- Commercial use — asset licenses forbid it (see §9)

---

## 4. Success Criteria

These are the numbers that go in the README. If they are not measured, the goal
is not met.

| ID | Criterion | Target | Measured by |
|----|-----------|--------|-------------|
| SC1 | NoW benchmark median error | Within 10% of published baseline | `eval/now_harness.py` |
| SC2 | End-to-end web latency, single image | < 500 ms on mid-range laptop GPU | Perf overlay, p50/p95 |
| SC3 | Webcam tracking framerate | ≥ 24 fps sustained, ≥ 30 fps target | Perf overlay |
| SC4 | Model download size | < 50 MB total, quantized | CI check |
| SC5 | Export round-trip | Loads + animates in 3 engines, 0 validator errors | `test_export_roundtrip.py` |
| SC6 | Cold start to first mesh | < 5 s on cable broadband | Manual, documented |

**Definition of done for the project:** a public URL a stranger can open, upload
a photo, and download a working asset — plus a README leading with a GIF and
the table above.

---

## 5. Architecture

```
                    ┌─────────────────────────┐
                    │   face3d-core (Python)  │
                    │                         │
   image ──────────▶│  detect → align →       │
                    │  encode → FLAME params  │──▶ params (~400 floats)
                    │         │               │
                    │         ▼               │
                    │  params → mesh →        │
                    │  rig → glTF export      │──▶ .gltf / .glb
                    └─────────────────────────┘
                         │              │
              ┌──────────┘              └──────────┐
              ▼                                    ▼
      ┌───────────────┐                   ┌─────────────────┐
      │ Blender add-on│                   │  ONNX export    │
      │   (Python)    │                   │       ↓         │
      └───────────────┘                   │  Web frontend   │
                                          │ (WebGPU/WASM +  │
                                          │   three.js)     │
                                          └─────────────────┘
```

### Key boundaries

- **`face3d-core` knows nothing about its consumers.** No Blender imports, no
  web concerns. Pure: bytes in, parameters and meshes out.
- **FLAME is frozen.** It is a fixed statistical basis, not a trainable
  component. Only the encoder is ever fine-tuned.
- **The encoder is swappable behind an interface.** DECA, SMIRK, and MICA all
  satisfy `Encoder.predict(image) -> FlameParams`. This is the escape hatch that
  makes the ML phase safe to fail (see §8, R2).
- **Text conditioning, if built, targets expression only.** An LLM parses natural
  language into a structured, named attribute schema; deterministic code maps
  that schema to FLAME parameters. The LLM never emits raw coefficients.

### Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Head model | FLAME 2020 | Frozen. CC BY, non-commercial |
| Encoder | DECA baseline, SMIRK if expressions weak | Swappable |
| Rasterizer | nvdiffrast, PyTorch3D fallback | PyTorch3D install is a known hazard |
| Landmarks | MediaPipe FaceMesh | FAN if matching DECA exactly |
| Segmentation | BiSeNet face-parsing | For masked photometric loss |
| Web runtime | ONNX Runtime Web, WebGPU + WASM fallback | |
| Web render | three.js | |
| Export | glTF 2.0 with morph targets | |
| CI | GitHub Actions | Runs eval harness + export tests |

---

## 6. Milestones

| # | Milestone | Weeks | Exit criterion |
|---|-----------|-------|----------------|
| M0 | Baseline reproduced | 1–2 | Published NoW numbers reproduced within noise |
| M1 | Asset pipeline | 3–4 | Rigged glTF opens and animates in Blender + three.js |
| M2 | ML contribution | 5–7 | Measured win over M0 baseline, **or** documented negative result |
| M3 | Web delivery | 8–10 | Live URL, webcam tracking hitting SC3 |
| M4 | Legibility | 11–12 | README, deployed demo, writeup published |

Blender add-on is scheduled inside M3 only if the web demo is on track; it is the
designated scope cut.

---

## 7. Backlog

Estimates are in points: **1** ≈ a session, **2** ≈ a week of evenings,
**3** ≈ multi-week. Priority: **P0** blocking, **P1** committed, **P2** if time.

### Epic A — Foundations (M0)

| ID | Story | Pri | Est | Acceptance |
|----|-------|-----|-----|------------|
| A1 | Register for FLAME + NoW, accept licenses | P0 | 1 | Assets downloaded, license terms recorded in §9 |
| A2 | Pin environment: PyTorch, CUDA, rasterizer | P0 | 2 | `make setup` works from clean clone; versions locked |
| A3 | Run DECA inference unmodified on 10 photos | P0 | 1 | Meshes produced, visually sane |
| A4 | Build NoW eval harness | P0 | 2 | `make eval` prints median/mean/std error |
| A5 | Reproduce published baseline numbers | P0 | 2 | Within noise of paper; **hard gate** |
| A6 | Assemble 30–50 image demo set, consented | P1 | 1 | Committed to repo, provenance documented |
| A7 | CI skeleton: lint, test, eval-on-subset | P1 | 1 | Green build on push |

> **Gate G-A:** Do not proceed past A5. A pipeline built on an unverified
> baseline produces numbers that mean nothing.

### Epic B — Core library & asset pipeline (M1)

| ID | Story | Pri | Est | Acceptance |
|----|-------|-----|-----|------------|
| B1 | Define `Encoder` interface; wrap DECA behind it | P0 | 2 | Second encoder can be added without touching callers |
| B2 | `image → FlameParams` public API, typed | P0 | 2 | Documented, type-hinted, unit tested |
| B3 | `FlameParams → mesh` with FLAME topology | P0 | 1 | Vertex count and ordering verified against template |
| B4 | Build armature: neck, jaw, eye joints | P0 | 2 | Joints articulate correctly in Blender |
| B5 | Emit expression basis as glTF morph targets | P0 | 3 | Blendshapes drivable by name in three.js |
| B6 | glTF/GLB writer with UVs and materials | P0 | 2 | Passes Khronos glTF-Validator, 0 errors |
| B7 | Golden-file export tests | P0 | 2 | Committed reference mesh; CI fails on drift |
| B8 | Round-trip tests: Blender, three.js, Unity | P1 | 2 | Documented load + animate in each; SC5 met |
| B9 | Texture/albedo extraction | P2 | 2 | UV texture exported alongside mesh |

> B5–B8 are the highest-signal work in the entire project. Research repos
> stop at B3.

### Epic C — ML contribution (M2)

Pick **exactly one** of C-alt-1/2/3 at start of week 5.

| ID | Story | Pri | Est | Acceptance |
|----|-------|-----|-----|------------|
| C1 | Self-supervised training loop: landmark + photometric + identity + reg losses | P0 | 3 | Loss curves logged; overfits a single batch |
| C2 | Face segmentation masks into photometric loss | P0 | 2 | Hair/glasses/hands excluded; ablation shows effect |
| C3 | Freeze backbone, train target branch only | P0 | 2 | Trains in < 48h on one GPU |
| C4 | Ablation table vs. M0 baseline | P0 | 2 | ≥ 3 configurations, same eval harness |
| **C-alt-1** | Expression refinement | — | 3 | Improved expression fidelity, measured |
| **C-alt-2** | Occlusion robustness | — | 3 | Error under synthetic occlusion improved |
| **C-alt-3** | Semantic shape directions (enables text) | — | 3 | Labeled attributes regressed to coefficients; N interpretable sliders |

> **Gate G-C (end of week 7):** No measured improvement → stop. Ship the
> baseline, write up the negative result with the ablation table, redirect
> remaining hours to Epic D. A well-engineered app with an honest failure
> analysis is a stronger artifact than a half-finished training run.

### Epic D — Web delivery (M3)

| ID | Story | Pri | Est | Acceptance |
|----|-------|-----|-----|------------|
| D1 | ONNX export + quantization | P0 | 2 | Parity vs. PyTorch within tolerance; SC4 met |
| D2 | ORT Web with WebGPU, WASM fallback | P0 | 2 | Runs on a machine with no WebGPU |
| D3 | Upload → mesh → three.js viewer | P0 | 2 | SC2 met |
| D4 | Download exported asset from browser | P0 | 1 | File opens in Blender |
| D5 | Inference in Web Worker | P0 | 2 | Main thread stays responsive during inference |
| D6 | Webcam loop with frame **dropping** (not queueing) | P0 | 3 | SC3 met; no unbounded latency growth under load |
| D7 | Perf overlay: detect / infer / render split, p50 + p95 | P1 | 1 | Visible in demo |
| D8 | Failure handling: no face, multi-face, profile, HEIC, 50MP, interrupted download | P1 | 2 | Each case has a defined, tested behavior |
| D9 | Blender add-on reusing `face3d-core` | P1 | 3 | Installs as extension; imports rigged head |
| D10 | Text→expression: LLM parses to attribute schema | P2 | 2 | Schema validated; deterministic mapping to params |

### Epic E — Legibility (M4)

| ID | Story | Pri | Est | Acceptance |
|----|-------|-----|-----|------------|
| E1 | Deploy demo to static host | P0 | 1 | Public URL, works on a stranger's machine |
| E2 | README: GIF above fold, results table, arch diagram | P0 | 2 | SC table filled with real numbers |
| E3 | License and data provenance page | P0 | 1 | §9 published |
| E4 | Technical writeup on hardest bug | P1 | 2 | Published; linked from README |
| E5 | 30s demo video | P2 | 1 | Embedded in README |

---

## 8. Risks

| ID | Risk | L | I | Mitigation | Kill criterion |
|----|------|---|---|------------|----------------|
| R1 | Rasterizer install hell (PyTorch3D/CUDA) | High | High | Pin versions in A2; nvdiffrast as primary | > 1 week lost → switch renderer |
| R2 | Fine-tune shows no improvement | Med | Med | Encoder is swappable; gate G-C | Week 7 → ship baseline, document |
| R3 | License approval delays | Med | High | Submit all requests in week 1 | > 2 weeks → substitute open model |
| R4 | WebGPU unavailable on target devices | Med | Med | WASM fallback is P0, not P2 | — |
| R5 | Quantization degrades quality unacceptably | Low | Med | Measure parity in D1 before building on it | Ship fp16 and eat the size |
| R6 | glTF morph targets don't survive engine import | Med | High | B7/B8 tests catch this early, not at demo time | Fall back to FBX |
| R7 | Scope creep into photorealism | Med | High | §3 non-goals are binding | — |
| R8 | License forbids shipping weights to browsers | **Certain** | High | Server-side demo until own encoder trained (§9) | — |

---

## 9. License & Data Register

Fill this in as assets are acquired. It is a deliverable, not bookkeeping.

| Asset | Source | License | Commercial | Redistributable | Acquired |
|-------|--------|---------|-----------|-----------------|----------|
| FLAME 2020 | flame.is.tue.mpg.de | Non-commercial research | No | No | ☑ |
| FLAME 2023 | same | Non-commercial research | No | No | ☑ |
| **FLAME 2023 Open** | same | **CC-BY-4.0** | **Yes** | **Yes** | ☐ verify variant |
| FLAME texture space | same | CC BY-NC-SA 4.0 | No | No | ☑ |
| NoW benchmark | MPI-IS | Research use | No | No | ☐ |
| DECA weights | DECA repo | Non-commercial research | No | **No** | ☐ |
| BFM albedo (`FLAME_albedo_from_BFM.npz`) | Separate BFM registration | Research | No | No | ☐ if albedo in scope |
| FFHQ (if used) | NVIDIA | CC BY-NC-SA 4.0 | No | — | ☐ |
| MediaPipe face landmarker | storage.googleapis.com/mediapipe-models | Apache-2.0 | Yes | Yes | ☑ |
| **MediaPipe selfie multiclass segmenter** | same | **Apache-2.0** | **Yes** | **Yes** | ☑ 2026-08-29 |
| Demo photos | Self + consented friends | Explicit consent on file | N/A | N/A | ☐ |

### The distribution constraint (added 2026-08-20)

DECA's license limits installation to machines the licensee owns or controls,
and forbids copying, sharing, or transferring the model apart from one archival
copy. **Shipping quantized DECA weights to visitors' browsers is distribution.**
Deriving an ONNX file does not change this.

Consequences:

- **Public demo must be server-side or pre-baked** while any DECA-derived weight
  is in the pipeline.
- **D1/D2 (on-device inference) are blocked** until the shipped encoder is one we
  trained ourselves against FLAME 2023 Open.
- This makes "retarget the encoder to FLAME 2023 Open" (C-alt-3 variant) not just
  an ML contribution but **the unlock for the entire client-side story.** Consider
  promoting it.
- Portfolio/CV use of a non-commercial model is grey but defensible as
  non-commercial education. Distribution is not grey. Treat them separately.
- **Action:** email MPI for written clarification. Written permission beats
  interpretation, and the reply belongs in this repo.

**Standing rules:**

1. **No scraping faces.** Face data is biometric data under GDPR and Illinois
   BIPA. Scraping converts a portfolio asset into a liability.
2. **Verify a dataset is still active before use.** MS-Celeb-1M and DukeMTMC were
   withdrawn; citing them signals you didn't check.
3. **This project is non-commercial and the README says so.** Being able to
   explain why, in an interview, is part of the value.

---

## 10. Open Questions

- **Q1.** DECA or SMIRK as the M0 baseline? Decide after A3 on expression
  quality. MICA is the alternative if identity accuracy matters more.
- **Q2.** Does the FLAME expression basis map cleanly enough onto ARKit
  blendshape names to be worth exposing that vocabulary? Affects D10.
- **Q3.** glTF morph targets vs. FBX for the Unity path — resolve in B8.
- **Q4.** Is C-alt-3 (semantic directions) too ambitious for 3 weeks? It is the
  most interesting but the least certain to produce a measurable win.

---

## Appendix — Week One Checklist

1. Submit FLAME registration
2. Submit NoW registration
3. Create repo, push skeleton, enable CI
4. Pin environment; get the rasterizer importing
5. Run DECA on one photo end to end
6. Draft the demo image set and obtain consent

Nothing in week one requires a GPU except step 5. Registrations are the
critical path — submit them before anything else.
