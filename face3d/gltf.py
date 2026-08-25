"""glTF 2.0 / GLB export — stories B4, B5, B6.

Turns FLAME output into an asset an engine can load and animate: a skinned mesh
with a neck/jaw/eye armature and the expression basis as named morph targets.
This is the step research repos stop short of, and it is where most of the
project's value sits.

Three things here are easy to get subtly wrong, so they are handled explicitly:

  Joint influences  glTF's JOINTS_0/WEIGHTS_0 are vec4, four influences per
                    vertex. FLAME has five joints and some vertices are
                    influenced by all five. The smallest influence is dropped
                    and the rest renormalised, rather than letting a vertex
                    silently lose weight and drift on animation.

  Inverse binds     A skin needs inverseBindMatrices, the transform taking each
                    vertex from model space into each joint's local space at
                    rest. FLAME's joints are positions with no rotation at rest,
                    so these are pure negative translations -- but they must be
                    present and column-major or the mesh collapses on load.

  Alignment         Every glTF bufferView must start on a multiple of its
                    component size, and both GLB chunks pad to 4 bytes. Getting
                    this wrong produces a file that some viewers open and others
                    reject, which is the worst possible failure mode.
"""

import base64
import json
import struct

import numpy as np

# glTF component types
FLOAT = 5126
UNSIGNED_INT = 5125
UNSIGNED_SHORT = 5123

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942


class _Buffer:
    """Accumulates binary data and hands back accessor indices."""

    def __init__(self):
        self.data = bytearray()
        self.views = []
        self.accessors = []

    def _pad(self, alignment=4):
        while len(self.data) % alignment:
            self.data.append(0)

    def add_bytes(self, raw):
        """Append opaque bytes (an embedded PNG) and return its bufferView index.

        No accessor: glTF images reference a bufferView directly.
        """
        self._pad(4)
        offset = len(self.data)
        self.data.extend(raw)
        self.views.append({"buffer": 0, "byteOffset": offset,
                           "byteLength": len(raw)})
        return len(self.views) - 1

    def add(self, array, comp_type, type_str, target=None, minmax=False):
        """Append an array, return its accessor index."""
        arr = np.ascontiguousarray(array)
        itemsize = arr.dtype.itemsize
        self._pad(itemsize)
        offset = len(self.data)
        self.data.extend(arr.tobytes())

        view = {"buffer": 0, "byteOffset": offset, "byteLength": arr.nbytes}
        if target is not None:
            view["target"] = target
        self.views.append(view)

        count = arr.shape[0] if arr.ndim > 1 else arr.size
        acc = {"bufferView": len(self.views) - 1, "componentType": comp_type,
               "count": int(count), "type": type_str}
        if minmax:
            flat = arr.reshape(count, -1)
            acc["min"] = [float(x) for x in flat.min(axis=0)]
            acc["max"] = [float(x) for x in flat.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def _top4_influences(weights):
    """(V,J) skinning weights -> (V,4) joint indices and normalised weights.

    FLAME has five joints; glTF allows four influences per vertex. Keeping the
    four largest and renormalising preserves the partition of unity, so a vertex
    cannot lose weight and drift when the skeleton moves.
    """
    v, j = weights.shape
    k = min(4, j)
    idx = np.argsort(-weights, axis=1)[:, :k]
    w = np.take_along_axis(weights, idx, axis=1)
    total = w.sum(axis=1, keepdims=True)
    w = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
    if k < 4:  # pad out to vec4
        idx = np.pad(idx, ((0, 0), (0, 4 - k)))
        w = np.pad(w, ((0, 0), (0, 4 - k)))
    return idx.astype(np.uint16), w.astype(np.float32)


def vertex_normals(verts, faces):
    """(V,3) smooth normals, area-weighted by the cross product magnitude.

    Without a NORMAL attribute glTF requires the viewer to compute FLAT
    per-face normals, so a 9,976-triangle head renders visibly faceted. Supply
    them.
    """
    v = np.asarray(verts, np.float32)
    f = np.asarray(faces, np.int64)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for c in range(3):
        np.add.at(n, f[:, c], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-12).astype(np.float32)


def build_gltf(verts, faces, joints, parents, skin_weights, joint_names,
               morph_targets=None, morph_names=None, uv=None, name="face",
               texture_png=None, normals=None):
    """Assemble the glTF JSON and its binary blob.

    verts (V,3), faces (F,3), joints (J,3) rest positions, parents (J,),
    skin_weights (V,J), morph_targets (M,V,3) as position DELTAS from verts.

    texture_png: raw PNG bytes for the baseColour map, embedded in the binary
    chunk so the GLB stays a single self-contained file. Requires `uv`.
    normals: (V,3); computed from the mesh when omitted.
    """
    buf = _Buffer()
    verts = np.asarray(verts, np.float32)
    faces = np.asarray(faces, np.uint32)
    joints = np.asarray(joints, np.float32)

    a_pos = buf.add(verts, FLOAT, "VEC3", target=34962, minmax=True)
    a_idx = buf.add(faces.reshape(-1), UNSIGNED_INT, "SCALAR", target=34963)
    attributes = {"POSITION": a_pos}

    if normals is None:
        normals = vertex_normals(verts, faces)
    attributes["NORMAL"] = buf.add(np.asarray(normals, np.float32), FLOAT, "VEC3",
                                   target=34962)

    if uv is not None:
        attributes["TEXCOORD_0"] = buf.add(np.asarray(uv, np.float32), FLOAT, "VEC2",
                                           target=34962)

    ji, jw = _top4_influences(np.asarray(skin_weights, np.float32))
    attributes["JOINTS_0"] = buf.add(ji, UNSIGNED_SHORT, "VEC4", target=34962)
    attributes["WEIGHTS_0"] = buf.add(jw, FLOAT, "VEC4", target=34962)

    primitive = {"attributes": attributes, "indices": a_idx, "mode": 4}

    materials, images, textures, samplers = [], [], [], []
    if texture_png is not None:
        if uv is None:
            raise ValueError("texture_png needs uv; without TEXCOORD_0 a "
                             "baseColorTexture has nothing to sample against")
        images.append({"bufferView": buf.add_bytes(texture_png),
                       "mimeType": "image/png", "name": "albedo"})
        # 9729/9987 = LINEAR / LINEAR_MIPMAP_LINEAR, 10497 = REPEAT
        samplers.append({"magFilter": 9729, "minFilter": 9987,
                         "wrapS": 10497, "wrapT": 10497})
        textures.append({"sampler": 0, "source": 0})
        # Skin is dielectric: metallic 0. Roughness high so the SH-lit albedo
        # is not given a specular sheen it was never estimated with.
        materials.append({
            "name": "skin",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 0.85,
            },
            "doubleSided": False,
        })
        primitive["material"] = 0

    if morph_targets is not None and len(morph_targets):
        targets, weights0 = [], []
        for d in np.asarray(morph_targets, np.float32):
            targets.append({"POSITION": buf.add(d, FLOAT, "VEC3", target=34962,
                                                minmax=True)})
            weights0.append(0.0)
        primitive["targets"] = targets

    # Inverse bind matrices: rest joints carry no rotation, so each is a pure
    # translation by -joint_position, written column-major as glTF requires.
    ibm = np.tile(np.eye(4, dtype=np.float32), (len(joints), 1, 1))
    ibm[:, 3, :3] = -joints           # column-major: translation in row 3
    a_ibm = buf.add(ibm.reshape(len(joints), 16), FLOAT, "MAT4")

    # Node 0 is the mesh; joints follow. Joint translations are relative to the
    # parent joint, which is what a glTF skeleton expects.
    nodes = [{"name": name, "mesh": 0, "skin": 0}]
    joint_node_ids = []
    for j, jname in enumerate(joint_names):
        t = joints[j] if parents[j] < 0 else joints[j] - joints[parents[j]]
        nodes.append({"name": jname, "translation": [float(x) for x in t]})
        joint_node_ids.append(len(nodes) - 1)
    for j, par in enumerate(parents):
        if par >= 0:
            nodes[joint_node_ids[par]].setdefault("children", []).append(
                joint_node_ids[j])

    root_joints = [joint_node_ids[j] for j, par in enumerate(parents) if par < 0]
    scene_nodes = [0] + root_joints

    mesh = {"name": name, "primitives": [primitive]}
    if morph_targets is not None and len(morph_targets):
        mesh["weights"] = weights0
        if morph_names:
            mesh["extras"] = {"targetNames": list(morph_names)}

    gltf = {
        "asset": {"version": "2.0", "generator": "face3d"},
        "scene": 0,
        "scenes": [{"nodes": scene_nodes}],
        "nodes": nodes,
        "meshes": [mesh],
        "skins": [{"inverseBindMatrices": a_ibm, "joints": joint_node_ids,
                   "skeleton": root_joints[0]}],
        "accessors": buf.accessors,
        "bufferViews": buf.views,
        "buffers": [{"byteLength": len(buf.data)}],
    }
    # Omitted entirely rather than left empty: the validator flags empty arrays.
    if materials:
        gltf["materials"] = materials
        gltf["images"] = images
        gltf["textures"] = textures
        gltf["samplers"] = samplers
    return gltf, bytes(buf.data)


def glb_bytes(gltf, blob):
    """Serialise a binary glTF to bytes. Both chunks pad to 4, as the spec says.

    Separate from write_glb because a server must be able to produce a GLB
    without touching the filesystem: uploaded faces are biometric data and the
    less that reaches disk the better.
    """
    gltf = dict(gltf)
    gltf["buffers"] = [{"byteLength": len(blob)}]
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    bin_pad = blob + b"\x00" * ((4 - len(blob) % 4) % 4)

    total = 12 + 8 + len(js) + 8 + len(bin_pad)
    return b"".join([
        struct.pack("<III", GLB_MAGIC, 2, total),
        struct.pack("<II", len(js), CHUNK_JSON), js,
        struct.pack("<II", len(bin_pad), CHUNK_BIN), bin_pad,
    ])


def write_glb(path, gltf, blob):
    """Write a binary glTF."""
    with open(path, "wb") as f:
        f.write(glb_bytes(gltf, blob))
    return path


def write_gltf(path, gltf, blob):
    """Write a .gltf with the buffer embedded as a data URI (easier to inspect)."""
    gltf = dict(gltf)
    gltf["buffers"] = [{"byteLength": len(blob),
                        "uri": "data:application/octet-stream;base64,"
                               + base64.b64encode(blob).decode("ascii")}]
    with open(path, "w") as f:
        json.dump(gltf, f, indent=1)
    return path
