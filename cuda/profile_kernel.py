import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face3d import assets
from face3d.render import raster_cuda
from face3d.geometry.flame_torch import FlameTorch
from face3d.render.pipeline import project

B, S = 8, 224  # the encoder's training batch and render size

fl = FlameTorch(assets.model_path("FLAME2023Open/flame2023_Open.pkl")).cuda()
g = torch.Generator().manual_seed(0)
shape = (torch.randn(B, fl.n_shape, generator=g) * 0.8).cuda()
expr = (torch.randn(B, fl.n_expr, generator=g) * 0.3).cuda()
with torch.no_grad():
    v, _ = fl(shape, expr, None)
    ndc = project(v, torch.tensor([[7.0, 0.0, 0.0]], device="cuda").expand(B, 3))

px = torch.stack([(ndc[..., 0] * .5 + .5) * S,
                  (.5 - ndc[..., 1] * .5) * S], -1).contiguous()
dep = ndc[..., 2].contiguous()
faces = fl.faces.contiguous().int()

ext = raster_cuda.extension()
assert ext is not None, raster_cuda.reason()

# Three launches: ncu is told to skip the first two, so it measures a warm one.
for _ in range(3):
    ext.assign_faces(px, dep, faces, S, S)
torch.cuda.synchronize()
print(f"launched assign_faces 3x at B={B} {S}x{S}")
