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

_CU = pathlib.Path(__file__).resolve().parents[2] / "cuda"

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
            import torch as _t
            _reason = (f"no CUDA toolkit: CUDA_HOME is None. torch was built "
                       f"against CUDA {_t.version.cuda}, so install a matching "
                       f"toolkit (winget install --id Nvidia.CUDA --version "
                       f"{_t.version.cuda}) -- and on Windows an MSVC toolset "
                       f"that version accepts")
            return None

        # Build for the device actually present. Compiling for the wrong
        # architecture still links and then fails at launch with "no kernel
        # image is available", which is a confusing way to find out.
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)

        # -lineinfo costs nothing at runtime and lets Nsight Compute map
        # stalls back to source lines; see cuda/profile.sh.
        flags = ["-O3", f"-arch=sm_{major}{minor}", "-lineinfo"]
        if os.name == "nt":
            # CUDA 13's CCCL headers refuse MSVC's traditional preprocessor:
            #   preprocessor.h: MSVC/cl.exe with traditional preprocessor is used
            # /Zc:preprocessor selects the standards-conformant one. Required
            # from CUDA 13 onward; harmless before it.
            flags += ["-Xcompiler", "/Zc:preprocessor"]
            # nvcc refuses host compilers newer than the ones it shipped
            # knowing about, and this machine has only MSVC 14.51 (VS 18) while
            # CUDA 12.8 expects 14.4x. Without this it stops at "unsupported
            # Microsoft Visual Studio version" before compiling anything. It is
            # a real override, not a formality: if PyTorch's headers then fail
            # to compile, install an older toolset rather than fighting it.
            flags.append("-allow-unsupported-compiler")

        _ext = load(
            name="face3d_raster",
            sources=[str(src)],
            extra_cuda_cflags=flags,
            extra_include_paths=[str(_CU)],
            verbose=False,
        )
        _reason = ""
    except Exception as e:                       # nvcc missing, no compiler, ...
        _ext = None
        _reason = f"{type(e).__name__}: {e}"
    return _ext
