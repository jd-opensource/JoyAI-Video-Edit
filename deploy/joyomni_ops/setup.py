"""Build script for joyomni_ops — a minimal, self-contained CUDA operator library
extracted from sgl-kernel for the JoyOmni V2V DiT pipeline.

Ops provided (no sgl-kernel / sglang dependency):
  - fused_qk_norm_rope_3d_paired : fused RMSNorm(q,k) + 3D RoPE   (bf16)
  - fused_norm_scale_shift       : fused LayerNorm/RMSNorm + adaLN modulate
  - rmsnorm                      : standalone RMSNorm
  - sgl_per_token_quant_fp8      : dynamic per-token bf16 -> fp8_e4m3 quant
  - fp8_scaled_mm                : FP8 per-token x per-channel scaled GEMM (cutlass)

GPU arch coverage is chosen from the local nvcc version:
  - always: sm_80, sm_89, sm_90
  - CUDA >= 12.8: also sm_100a (B200) and sm_120a (RTX PRO 6000 / RTX 5090)
So building on a CUDA 12.8+ toolchain automatically yields Blackwell support.
"""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

THIS_DIR = Path(__file__).parent.resolve()


def _cuda_version():
    """(major, minor) of the nvcc that torch will use, or (0, 0) if unknown."""
    from torch.utils.cpp_extension import CUDA_HOME
    import subprocess
    try:
        nvcc = os.path.join(CUDA_HOME or "/usr/local/cuda", "bin", "nvcc")
        out = subprocess.check_output([nvcc, "--version"], text=True)
        for line in out.splitlines():
            if "release" in line:
                tok = line.split("release")[1].strip().split(",")[0]  # "12.6"
                mj, mn = tok.split(".")[:2]
                return int(mj), int(mn)
    except Exception:
        pass
    return (0, 0)


def _gencodes():
    """-gencode flags. Blackwell (sm_100a/sm_120a) needs nvcc >= 12.8."""
    mj, mn = _cuda_version()
    # Allow override, e.g. JOYOMNI_OPS_CUDA_ARCHS="90;120a"
    override = os.environ.get("JOYOMNI_OPS_CUDA_ARCHS")
    if override:
        flags = []
        for a in override.split(";"):
            a = a.strip()
            if not a:
                continue
            code = f"sm_{a}"
            arch = f"compute_{a}"
            flags += [f"-gencode=arch={arch},code={code}"]
        return flags
    flags = [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ]
    if (mj, mn) >= (12, 8):
        flags += [
            "-gencode=arch=compute_100a,code=sm_100a",
            "-gencode=arch=compute_120a,code=sm_120a",
            # PTX fallback so newer archs (e.g. sm_121) can JIT.
            "-gencode=arch=compute_120,code=compute_120",
        ]
    else:
        # No SASS for sm_100/sm_120 on this toolchain; embed sm_90 PTX so the
        # driver can JIT for Blackwell (sm_100/sm_120) at load time. Correctness
        # only — for tuned Blackwell SASS, build on CUDA >= 12.8.
        flags += ["-gencode=arch=compute_90,code=compute_90"]
        print(
            f"[joyomni_ops] nvcc {mj}.{mn} < 12.8: SASS sm_80/89/90 + sm_90 PTX "
            f"(JIT fallback for Blackwell). Build on CUDA >= 12.8 for native sm_100a/sm_120a."
        )
    return flags


SOURCES = [
    "csrc/pybind.cpp",
    "csrc/fused_qknorm_rope_3d_kernel.cu",
    "csrc/fused_norm_scale_shift.cu",
    "csrc/rmsnorm.cu",
]

# FP8 GEMM needs cutlass (and CUDA >= 12.8 for Blackwell). Set JOYOMNI_OPS_NO_FP8=1
# to build only the 3 light kernels (no cutlass required).
NO_FP8 = os.environ.get("JOYOMNI_OPS_NO_FP8", "").lower() in {"1", "true", "yes", "on"}
if not NO_FP8:
    SOURCES += ["csrc/per_token_quant_fp8.cu", "csrc/fp8_gemm.cu"]

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
] + _gencodes()

cxx_flags = ["-O3", "-std=c++17"]
if NO_FP8:
    nvcc_flags.append("-DJOYOMNI_OPS_NO_FP8")
    cxx_flags.append("-DJOYOMNI_OPS_NO_FP8")

include_dirs = [str(THIS_DIR / "include")]

# cutlass (for fp8_scaled_mm). Point JOYOMNI_OPS_CUTLASS_DIR at a cutlass checkout
# (commit dcf215af matches the reference build).
cutlass_dir = os.environ.get("JOYOMNI_OPS_CUTLASS_DIR")
if not NO_FP8:
    if cutlass_dir:
        include_dirs += [
            os.path.join(cutlass_dir, "include"),
            os.path.join(cutlass_dir, "tools", "util", "include"),
        ]
    else:
        print("[joyomni_ops] JOYOMNI_OPS_CUTLASS_DIR not set — fp8_gemm.cu will fail to "
              "compile. Set it, or build with JOYOMNI_OPS_NO_FP8=1.")

setup(
    name="joyomni_ops",
    version="0.1.0",
    description="Minimal self-contained CUDA ops for the JoyOmni V2V DiT pipeline",
    packages=["joyomni_ops"],
    ext_modules=[
        CUDAExtension(
            name="joyomni_ops._C",
            sources=SOURCES,
            include_dirs=include_dirs,
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
