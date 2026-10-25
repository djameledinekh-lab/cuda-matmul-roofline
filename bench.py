"""Build the CUDA kernels, check them against torch.matmul, benchmark them and draw the Roofline.

Usage (on a machine or Colab runtime with an NVIDIA GPU):
    python bench.py                    # default sizes, COARSE=2
    python bench.py --coarse 4         # try another coarsening factor
"""
import argparse
import pathlib

import pandas as pd
import torch
from torch.utils.cpp_extension import load

import roofline

ROOT = pathlib.Path(__file__).resolve().parent


def build(coarse: int):
    return load(
        name=f"matmul_kernels_c{coarse}",
        sources=[str(ROOT / "csrc" / "matmul_kernels.cu")],
        extra_cuda_cflags=["-O3", f"-DCOARSE={coarse}"],
        verbose=False,
    )


def time_ms(fn, A, B, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn(A, B)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(A, B)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def check_correctness(kernels: dict) -> None:
    # 300 is not a multiple of 32 or 64, so the boundary checks are exercised too.
    A = torch.randn(300, 300, device="cuda")
    B = torch.randn(300, 300, device="cuda")
    ref = torch.matmul(A, B)
    for name, fn in kernels.items():
        if name == "cuBLAS":
            continue
        err = (fn(A, B) - ref).abs().max().item()
        status = "OK" if err < 1e-2 else "MISMATCH"
        print(f"  {name:10s} max |error| vs torch.matmul = {err:.2e}  {status}")
        if status != "OK":
            raise SystemExit(f"{name} gives wrong results, fix it before benchmarking")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--coarse", type=int, default=2, help="outputs per thread per dimension (2 or 4)")
    parser.add_argument("--max-naive", type=int, default=2048, help="skip the naive kernel above this size")
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. On Colab: Runtime > Change runtime type > T4 GPU")
    torch.backends.cuda.matmul.allow_tf32 = False  # keep cuBLAS in true FP32, like our kernels

    device = torch.cuda.get_device_name(0)
    spec = roofline.spec_for(device)
    print(f"GPU: {device} | peak {spec.fp32_tflops} TFLOPS FP32 | {spec.dram_gbs} GB/s | ridge {spec.ridge_point:.1f} FLOP/B")

    print(f"Compiling kernels (COARSE={args.coarse}), about a minute the first time...")
    ext = build(args.coarse)
    kernels = {
        "naive": ext.matmul_naive,
        "tiled": ext.matmul_tiled,
        "coarsened": ext.matmul_coarsened,
        "cuBLAS": torch.matmul,
    }
    print("Correctness:")
    check_correctness(kernels)

    rows = []
    for N in args.sizes:
        A = torch.randn(N, N, device="cuda")
        B = torch.randn(N, N, device="cuda")
        flops = 2 * N**3
        times = {name: time_ms(fn, A, B) for name, fn in kernels.items()
                 if not (name == "naive" and N > args.max_naive)}
        for name, ms in times.items():
            tflops = flops / (ms * 1e-3) / 1e12
            rows.append({
                "kernel": name,
                "N": N,
                "time_ms": ms,
                "tflops": tflops,
                "ai": roofline.arithmetic_intensity(name, N, args.coarse),
                "pct_of_peak": 100 * tflops / spec.fp32_tflops,
                "pct_of_cublas": 100 * times["cuBLAS"] / ms,
            })
        print(f"  N={N} done")

    df = pd.DataFrame(rows)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{spec.name.split()[0].lower()}_c{args.coarse}"
    df.to_csv(out / f"results_{tag}.csv", index=False)
    (out / f"results_{tag}.md").write_text(df.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
    roofline.plot(df, spec, str(out / f"roofline_{tag}.png"))
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved results and plot to {out}/")


if __name__ == "__main__":
    main()
