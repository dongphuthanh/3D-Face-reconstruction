"""Optional CUDA rasteriser, loaded on demand with a graceful fallback.

face3d.render._assign_faces is 94% of rasterize and rasterize is ~25% of an
encoder training step. Measured on an RTX 5070 at the training configuration
(batch 8, 224px, 9976 faces): 15.3 ms in PyTorch against 0.26 ms for the CUDA
kernel in cuda/raster_kernel.cuh, so a training step should drop by roughly a
quarter.

JIT-compiled rather than shipped prebuilt. A CUDA extension is tied to the exact
CUDA and Python it was built against, so a wheel would be wrong for anyone whose
toolchain differs -- and this repo has three of them in play already (Windows
torch on cu128, WSL nvcc 13.3, WSL python 3.14 with no torch). torch.utils
.cpp_extension.load compiles on first use and caches, which costs about a minute
once and nothing after.

FAILING TO LOAD IS NORMAL AND MUST BE SILENT-ISH. Most machines running this
have no nvcc. The import must not raise, must not print a wall of text, and must
leave the PyTorch path working -- so every failure is caught, recorded in
`reason()`, and turned into None. Ask for the reason when you want it; do not
make every unrelated user read it.
"""

import os
import pathlib

_CU = pathlib.Path(__file__).resolve().parents[1] / "cuda"

_ext = None
_reason = "not attempted"
_tried = False


def reason() -> str:
    """Why the CUDA path is unavailable, if it is. Empty when it loaded."""
    return "" if _ext is not None else _reason


def available() -> bool:
    return extension() is not None


def extension():
    """The compiled module, or None. Compiles on first call, then caches.

    Set FACE3D_NO_CUDA_RASTER=1 to force the PyTorch path -- useful when
    comparing the two, and the only way to be sure which one a benchmark ran.
    """
    global _ext, _reason, _tried
    if _tried:
        return _ext
    _tried = True

    if os.environ.get("FACE3D_NO_CUDA_RASTER"):
        _reason = "disabled by FACE3D_NO_CUDA_RASTER"
        return None

    try:
        import torch
        if not torch.cuda.is_available():
            _reason = "no CUDA device"
            return None

        src = _CU / "raster_ext.cu"
        if not src.exists():
            _reason = f"missing {src}"
            return None

        from torch.utils.cpp_extension import CUDA_HOME, load
        if CUDA_HOME is None:
            _reason = ("no CUDA toolkit (torch.utils.cpp_extension.CUDA_HOME is "
                       "None) -- nvcc is needed to build the extension, and it "
                       "must match the CUDA torch was built against")
            return None

        # Build for the device actually present. Compiling for the wrong
        # architecture still links and then fails at launch with "no kernel
        # image is available", which is a confusing way to find out.
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)

        _ext = load(
            name="face3d_raster",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", f"-arch=sm_{major}{minor}"],
            extra_include_paths=[str(_CU)],
            verbose=False,
        )
        _reason = ""
    except Exception as e:                       # nvcc missing, no compiler, ...
        _ext = None
        _reason = f"{type(e).__name__}: {e}"
    return _ext
