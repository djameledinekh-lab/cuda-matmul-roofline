"""Analytical Roofline model for the matmul kernels in csrc/matmul_kernels.cu.

Arithmetic intensity (AI) here is FLOPs divided by bytes moved between DRAM and the SM,
derived from each kernel's access pattern. It ignores L1/L2 cache hits, so it is a
lower bound on the real AI of the naive kernel and a model, not a measurement.
"""
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BYTES_PER_FLOAT = 4
TILE_TILED = 32   # TILE_SIZE in matmul_tiled_kernel
TILE_COARSE = 64  # TILE_C in matmul_coarsened_kernel


@dataclass(frozen=True)
class GpuSpec:
    name: str
    fp32_tflops: float  # peak FP32 on CUDA cores (no Tensor Cores)
    dram_gbs: float     # peak DRAM bandwidth

    @property
    def ridge_point(self) -> float:
        """FLOP/byte above which a kernel can become compute-bound."""
        return self.fp32_tflops * 1000 / self.dram_gbs


# NVIDIA datasheet values.
SPECS = {
    "T4": GpuSpec("Tesla T4", 8.1, 320.0),
    "A100": GpuSpec("A100 40GB", 19.5, 1555.0),
}


def spec_for(device_name: str) -> GpuSpec:
    for key, spec in SPECS.items():
        if key in device_name:
            return spec
    raise ValueError(f"No spec for '{device_name}': add its FP32 peak and DRAM bandwidth to SPECS in roofline.py")


def ai_naive(N: int) -> float:
    # Each FMA (2 FLOPs) reads one float of A and one of B from DRAM: 2 / (2 * 4).
    return 2 / (2 * BYTES_PER_FLOAT)


def ai_tiled(N: int, tile: int = TILE_TILED) -> float:
    # A tile x tile output block loads 2 * tile * N floats and does 2 * tile^2 * N FLOPs.
    return tile / BYTES_PER_FLOAT


def ai_cublas_ideal(N: int) -> float:
    # Ideal reuse: A and B read once, C written once -> 2N^3 / (3 * N^2 * 4).
    return 2 * N**3 / (3 * N**2 * BYTES_PER_FLOAT)


def ai_coarsened(N: int, tile: int = TILE_COARSE, coarse: int = 2) -> float:
    return tile/4

def arithmetic_intensity(kernel: str, N: int, coarse: int) -> float:
    if kernel == "naive":
        return ai_naive(N)
    if kernel == "tiled":
        return ai_tiled(N)
    if kernel == "coarsened":
        return ai_coarsened(N, coarse=coarse)
    if kernel == "cuBLAS":
        return ai_cublas_ideal(N)
    raise ValueError(f"Unknown kernel '{kernel}'")


STYLE = {
    "naive": ("#C2410C", "o"),
    "tiled": ("#B7791F", "s"),
    "coarsened": ("#4C51BF", "D"),
    "cuBLAS": ("#047857", "^"),
}


def plot(df, spec: GpuSpec, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ai = np.logspace(-1.5, 3.3, 300)
    roof = np.minimum(ai * spec.dram_gbs / 1000, spec.fp32_tflops)
    ax.plot(ai, roof, color="black", linewidth=1.5, label=f"Roofline ({spec.name}, FP32)")
    ax.axvline(spec.ridge_point, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(spec.ridge_point * 1.1, roof.min() * 1.5, f"ridge point = {spec.ridge_point:.1f} FLOP/B", fontsize=9, color="gray")

    for kernel, sub in df.groupby("kernel", sort=False):
        color, marker = STYLE.get(kernel, ("gray", "x"))
        ax.scatter(sub["ai"], sub["tflops"], color=color, marker=marker, s=70, zorder=5,
                   edgecolors="white", linewidths=0.5, label=kernel)
        for _, r in sub.iterrows():
            ax.annotate(f"N={r['N']}", (r["ai"], r["tflops"]), textcoords="offset points",
                        xytext=(6, -3), fontsize=7, color=color)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity, analytical (FLOP/byte)")
    ax.set_ylabel("Measured performance (TFLOPS)")
    ax.set_title(f"FP32 matmul on {spec.name}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
