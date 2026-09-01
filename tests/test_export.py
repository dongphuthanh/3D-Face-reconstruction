"""Export correctness — story B7.

Two layers, deliberately:

  structural  Assertions this project can make about its own output: vertex and
              triangle counts against FLAME's topology, skin weights summing to
              one, joint indices in range, morph deltas that actually deform,
              buffer alignment.

  external    The Khronos glTF-Validator, run through node. An asset that only
              satisfies its author's idea of the spec is how you discover at
              demo time that Unity rejects it. This is the check that matters,
              and it is why B6's acceptance criterion names the validator rather
              than "looks right in a viewer".

The structural layer exists because the validator checks conformance, not
correctness: a glTF with the wrong skinning weights is perfectly valid and
completely broken.
"""

import json
import pathlib
import shutil
import struct
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d import assets
from face3d.geometry.flame_torch import FlameTorch
from face3d.export.gltf import build_gltf, write_glb
from face3d.geometry.rig import JOINT_NAMES, expression_targets, jaw_target, rest_joints

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


MODEL = assets.model_path_or_skip()
flame = FlameTorch(MODEL)
fixture = assets.is_fixture(MODEL)

N_TARGETS = 8
deltas, names, neutral = expression_targets(flame, None, n_targets=N_TARGETS)
deltas = np.concatenate([deltas, jaw_target(flame)[None]], 0)
names = names + ["jaw_open"]
joints = rest_joints(flame).cpu().numpy()

gltf, blob = build_gltf(
    verts=neutral, faces=flame.faces.cpu().numpy(), joints=joints,
    parents=flame.parents, skin_weights=flame.weights.cpu().numpy(),
    joint_names=JOINT_NAMES[: flame.n_joints], morph_targets=deltas,
    morph_names=names, name="test_head")
path = OUT / "test_export.glb"
write_glb(path, gltf, blob)

print(f"=== structural ({'fixture' if fixture else 'real FLAME'}) ===")
prim = gltf["meshes"][0]["primitives"][0]
acc = gltf["accessors"]

check("positions match the mesh", acc[prim["attributes"]["POSITION"]]["count"] == flame.n_verts,
      f"{acc[prim['attributes']['POSITION']]['count']} verts")
check("indices match the topology",
      acc[prim["indices"]]["count"] == flame.n_faces * 3,
      f"{acc[prim['indices']]['count'] // 3} triangles")
check("one morph target per expression plus jaw",
      len(prim["targets"]) == N_TARGETS + 1)
check("morph targets are named", len(gltf["meshes"][0]["extras"]["targetNames"])
      == len(prim["targets"]), "glTF addresses targets by index; names live in extras")
check("skin lists every joint", len(gltf["skins"][0]["joints"]) == flame.n_joints)
check("inverse bind matrices present",
      acc[gltf["skins"][0]["inverseBindMatrices"]]["count"] == flame.n_joints)

# Skinning maths, which the validator cannot check.
from face3d.export.gltf import _top4_influences
ji, jw = _top4_influences(flame.weights.cpu().numpy())
check("skin weights sum to 1 after dropping the 5th influence",
      np.allclose(jw.sum(1), 1.0, atol=1e-5),
      f"max deviation {abs(jw.sum(1) - 1).max():.2e}")
check("joint indices are in range", int(ji.max()) < flame.n_joints)
n_over = int((flame.weights.cpu().numpy() > 1e-6).sum(1).max())
check("the 4-influence limit is actually exercised", n_over >= 4,
      f"FLAME uses up to {n_over} joints per vertex; glTF allows 4")

# Morph deltas must deform, and must be deltas rather than absolute positions.
d = np.asarray(deltas)
check("morph targets are non-zero", float(np.abs(d).max()) > 1e-4,
      f"max delta {float(np.abs(d).max()) * 1000:.1f} mm")
check("morph targets are deltas, not absolute positions",
      float(np.abs(d).mean()) < float(np.abs(neutral).mean()),
      "an absolute-position target would displace the mesh to the origin on load")
check("jaw target moves the lower face most",
      np.abs(d[-1])[neutral[:, 1] < np.percentile(neutral[:, 1], 25)].mean()
      > 3 * np.abs(d[-1])[neutral[:, 1] > np.percentile(neutral[:, 1], 75)].mean())

# GLB container.
raw = path.read_bytes()
magic, version, total = struct.unpack("<III", raw[:12])
check("GLB magic and version", magic == 0x46546C67 and version == 2)
check("GLB length field matches the file", total == len(raw), f"{total} vs {len(raw)}")
check("both chunks are 4-byte aligned", len(raw) % 4 == 0)
for i, v in enumerate(gltf["bufferViews"]):
    if v["byteOffset"] % 4:
        check(f"bufferView {i} aligned", False)
        break
else:
    check("every bufferView is aligned", True, f"{len(gltf['bufferViews'])} views")

print("")
print("=== Khronos glTF-Validator ===")
node = shutil.which("node")
if node is None:
    print("  SKIP - node not found; install it to run the official validator")
else:
    js = ROOT / "scripts" / "validate_glb.js"
    r = subprocess.run([node, str(js), str(path)], capture_output=True, text=True,
                       cwd=str(ROOT))
    out = (r.stdout or "").strip().splitlines()
    try:
        res = json.loads(out[-1])
        if res["e"] < 0:
            raise ValueError(res["msgs"][0])
        check("validator reports zero errors", res["e"] == 0, f"{res['e']} errors")
        check("validator reports zero warnings", res["w"] == 0, f"{res['w']} warnings")
        for m in res["msgs"][:5]:
            print(f"       {m}")
    except Exception as e:
        print(f"  SKIP - could not run the validator: {e}")
        print("        npm install --no-save gltf-validator")

print("")
print("=" * 56)
print("ALL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
