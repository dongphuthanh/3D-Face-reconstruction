#!/usr/bin/env bash
# Profile assign_faces with Nsight Compute.
#
#   ./cuda/profile.sh            summary to the terminal
#   ./cuda/profile.sh report     also write cuda/assign_faces.ncu-rep for the GUI
#
# Delegates to cuda/_profile_env.bat, which holds the machine-specific MSVC,
# CUDA and Nsight paths -- torch's JIT loader shells out to `where cl` on every
# load(), so ncu has to inherit a vcvars environment or the extension never
# loads and there is no kernel to profile.
#
# Must run ELEVATED, or ncu fails with ERR_NVGPUCTRPERM. The permanent fix is
# RmProfilingAdminOnly = 0 under
# HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak, which is
# already set -- but the driver reads it only at boot, and this box has not
# rebooted since.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
exec cmd.exe //c ".\_profile_env.bat ${1:-}"
