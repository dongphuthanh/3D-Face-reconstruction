# Project Context — Read Before Writing Code

Companion to `face3d-implementation-plan.md`. That document is the **what**; this
one is the **why and the current state**. It records decisions already made, so
they don't get relitigated or silently reversed.

---

## Current state

**Downloaded:** FLAME 2020, FLAME 2023, FLAME texture space (MPI).

**Not yet obtained:**

- `flame2023_Open.pkl` — verify which 2023 variant is on disk. The Open variant
  is the license-critical one (see Decision 4).
- NoW benchmark — **critical path.** Registration is slow and gates M0's exit
  criterion. Submit before anything else.
- DECA `deca_model.tar` — lives on the DECA repo's Google Drive, not the FLAME
  portal. ~400 MB.
- `FLAME_albedo_from_BFM.npz` — only if albedo enters scope. Requires a separate
  Basel Face Model registration. The MPI texture space already downloaded is
  **not** what DECA expects.

**Nothing has been built yet.** No repo, no environment, no code.

---

## Decisions already made

### 1. FLAME is frozen; only the encoder is trained

FLAME is a fixed statistical basis (PCA over ~33k scans + linear blend skinning).
Retraining it needs thousands of 3D scans. It is never a trainable component.
The learnable part is the encoder: the network mapping pixels → FLAME
coefficients. Any proposal to "fine-tune FLAME" is a misunderstanding.

### 2. Build on FLAME 2020, not 2023, for the baseline

DECA, SMIRK, EMOCA and the surrounding ecosystem were trained against FLAME
2020. Coefficients are meaningless outside the basis they were fit to — feeding
2020-trained coefficients into a 2023 basis produces a face that is subtly wrong
(especially the eye region, which 2023 revised) while still looking like a face.
That failure mode is expensive to debug.

Corroborating evidence: MPI shipped explicit conversion code just to translate
expression parameters between FLAME 2023 and FLAME 2023 Open. If two variants of
the same year need translation, 2020 → 2023 certainly does.

Second reason: published NoW numbers were produced with 2020. Changing the basis
means a discrepancy can't be attributed — bug or basis? That destroys the M0
gate, which is the only trustworthy reference point in the project.

### 3. Encoder sits behind an interface

`Encoder.predict(image) -> FlameParams`. DECA, SMIRK, and MICA all satisfy it.
This is a deliberate escape hatch: it makes the ML phase safe to fail and makes
Decision 4 executable without rewriting the pipeline.

### 4. The license constraint drives the architecture

**This is the most important thing in this document.**

DECA's license restricts installation to machines the licensee owns or controls,
and forbids copying, sharing, distributing, or transferring the model except for
one archival copy. Shipping quantized DECA weights to a visitor's browser is
distribution to a third-party machine. Converting to ONNX does not change this —
a derived checkpoint is still the model.

Therefore:

- The public demo is **server-side or pre-baked** for as long as any
  DECA-derived weight is in the pipeline.
- Client-side inference (plan items D1/D2) is **blocked** until the shipped
  encoder is one we trained ourselves.
- FLAME 2023 Open is CC-BY-4.0 — the only variant that is both commercially
  usable and redistributable. Training our own encoder against it is what
  unlocks the entire on-device story.
- Portfolio/CV use of a non-commercial model is grey but defensible as
  non-commercial education. **Distribution is not grey.** Keep the two questions
  separate; don't let a comfortable answer on the first one leak into the second.

### 5. Text conditioning targets expression, never identity

FLAME's ~300 shape parameters are unlabeled PCA axes. Component 47 does not mean
"wide jaw." An LLM asked to emit shape coefficients will produce confident
nonsense. Expression and pose (jaw, neck, eyes) are semantically meaningful and
map reasonably onto FACS/ARKit-style vocabulary.

So: LLM parses natural language into a **structured, named attribute schema**;
deterministic code maps that schema to FLAME parameters. The LLM is a parser, not
a parameter generator. Identity always comes from an uploaded image.

### 6. Bounded ML, heavy engineering

Training a face model from scratch is GPU-weeks a reviewer cannot evaluate. The
engineering — abstraction boundaries, export correctness, real-time constraints,
failure handling — is both tractable and the thing being assessed. Epic B
(rigged glTF export with round-trip tests) is the highest-value work in the
project. Research repos stop before it.

---

## Known hazards

| Hazard | Symptom | Fix |
|--------|---------|-----|
| FLAME pickles are Python 2 | `pickle.load` raises `UnicodeDecodeError`; reads as corrupt download | `encoding='latin1'` |
| PyTorch3D install | Hours lost to CUDA/torch version mismatch | Prefer nvdiffrast; pin everything |
| FLAME 2020 portal | "Access Denied" on 2020 while 2019/2023 work | Known portal quirk, not your account |
| Wrong texture asset | DECA can't find albedo | DECA wants `FLAME_albedo_from_BFM.npz`, not MPI texture space |
| Silent basis mismatch | Expressions subtly off, eyes worst | Verify FLAME version matches the checkpoint |

---

## Immediate next steps

1. Submit NoW registration. Critical path — do this first.
2. Verify which FLAME 2023 variant is on disk; get `flame2023_Open.pkl` if not.
3. Create the repo, push a skeleton, enable CI.
4. **Smoke-test FLAME alone before touching DECA.** Load `generic_model.pkl` via
   FLAME_PyTorch, zeros for shape and expression, save the mesh, open it. Then
   perturb one shape coefficient and confirm the face changes.
   _Rationale: isolates "is the model loading" from "is DECA working."
   Debugging both at once is miserable._
5. Pin environment (torch, CUDA, nvdiffrast).
6. Only then: download `deca_model.tar` and run DECA on one photo.
7. Email MPI for written license clarification; commit the reply to the repo.

---

## Hard gates

Both are in the plan. Neither is advisory.

- **G-A (end wk 2):** do not build on an unreproduced baseline. Numbers derived
  from an unverified starting point mean nothing.
- **G-C (end wk 7):** no measured improvement from the ML work → stop, ship the
  baseline, write up the negative result with its ablation table, redirect hours
  to the app. A well-engineered app with an honest failure analysis is a stronger
  artifact than a half-finished training run.

---

## Non-negotiables

- **No scraping faces.** Biometric data under GDPR and Illinois BIPA. Converts a
  portfolio asset into a liability.
- **Verify a dataset is still active before use.** MS-Celeb-1M and DukeMTMC were
  withdrawn.
- **Every asset gets a row in the license register** (plan §9) before it enters
  the pipeline.
- **Consent on file for every demo photo**, documented in the repo.
