"""Standalone driver for the candidate_v6 Mamba2 SSD Pallas kernel.

Self-contained: needs only ``candidate_v6.py`` next to it. Generates a
deterministic jagged workload, runs the kernel, optionally checks against
an inline fp32 reference, and prints median latency.

Default workload is the first 64 sequences from
``mamba2_ssd-msl1024-b512-sp0.95-a2-h128p64n128-bf16`` (n_seqs=64,
total tokens T=19120, H=128, P=64, N=128, chunk=256) — same length
distribution and tensor_init_seed as the project's b512 case, just
truncated to fit in one v6e chip's HBM.

Usage::

    # default workload + correctness check + timing
    python3 main.py

    # supply a custom case JSON (same schema as cases_mamba2_ssd/*.json)
    python3 main.py --case path/to/case.json

    # skip the fp32 reference (it's ~5x slower than the kernel itself)
    python3 main.py --no-correctness

    # tweak timing
    python3 main.py --warmup 5 --iters 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from candidate_v6 import ssd_candidate_jit


ATOL = 5e-2
RTOL = 5e-2


# Built-in workload: first 64 sequences of the project's
# mamba2_ssd-msl1024-b512-sp0.95-a2-h128p64n128-bf16 case (the full
# b512 version is ~180k tokens and OOMs on a single v6e chip).
# Same length distribution and tensor_init_seed as the project case,
# so this is a deterministic subset, not a fresh random workload.
DEFAULT_CASE = {
    "case_id": "mamba2_ssd-msl1024-b64-sp0.95-a2-h128p64n128-bf16 (built-in subset)",
    "heads": 128,
    "head_dim": 64,
    "state_dim": 128,
    "chunk_size": 256,
    "dtype": "bfloat16",
    "tensor_init_seed": 4113536370,
    "seq_lengths": [
        184, 94, 6, 604, 616, 511, 233, 56, 124, 113, 130, 872, 23, 550,
        959, 73, 22, 112, 817, 40, 413, 47, 96, 2, 443, 714, 33, 463,
        78, 618, 211, 936, 206, 10, 3, 864, 16, 30, 655, 259, 963, 688,
        3, 925, 605, 35, 57, 98, 615, 70, 1, 74, 623, 39, 2, 72, 2, 32,
        1, 879, 130, 48, 525, 397,
    ],
}


def _resolve_dtype(name: str):
    if name in ("bfloat16", "bf16"):
        return jnp.bfloat16
    if name in ("float16", "fp16"):
        return jnp.float16
    if name in ("float32", "fp32"):
        return jnp.float32
    raise ValueError(f"unknown dtype: {name}")


def materialize_case(case: dict):
    """Materialize (x, dt, A_log, B, C, cu_seqlens) for a case dict.

    Tensor init recipe matches the project's benchmark.py exactly:
      x      ~ N(0, 1)                       bf16
      dt     ~ softplus(0.5 * N(0, 1))       bf16   (already softplus-applied)
      A_log  ~ U(-4, -0.5)                   f32
      B, C   ~ N(0, 0.1)                     bf16
    """
    H = int(case["heads"])
    P = int(case["head_dim"])
    N = int(case["state_dim"])
    dtype = _resolve_dtype(case["dtype"])
    seed = int(case["tensor_init_seed"])

    if "seq_offsets" in case:
        cu = np.asarray(case["seq_offsets"], dtype=np.int32)
    else:
        lens = np.asarray(case["seq_lengths"], dtype=np.int32)
        cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    T = int(cu[-1])

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, H, P), dtype=np.float32)
    dt_raw = rng.standard_normal((T, H), dtype=np.float32) * 0.5
    dt = np.log1p(np.exp(dt_raw))
    A_log = rng.uniform(-4.0, -0.5, size=(H,)).astype(np.float32)
    B = rng.standard_normal((T, H, N), dtype=np.float32) * 0.1
    C = rng.standard_normal((T, H, N), dtype=np.float32) * 0.1

    return (
        jnp.asarray(x).astype(dtype),
        jnp.asarray(dt).astype(dtype),
        jnp.asarray(A_log),
        jnp.asarray(B).astype(dtype),
        jnp.asarray(C).astype(dtype),
        jnp.asarray(cu),
    )


def _reference_f32(x, dt, A_log, B, C, cu) -> jax.Array:
    """fp32 ideal Mamba2 SSD reference (segment-isolated scan).

    Padding tokens contribute zero (a==0 at segment boundaries), so this
    is the same reference the project uses to gate correctness at
    atol=rtol=5e-2.
    """
    T, H, P = x.shape
    N = B.shape[-1]
    x32 = x.astype(jnp.float32)
    dt32 = dt.astype(jnp.float32)
    B32 = B.astype(jnp.float32)
    C32 = C.astype(jnp.float32)

    a = jnp.exp(A_log[None, :] * dt32)
    idx = jnp.arange(T, dtype=cu.dtype)
    seg_id = jnp.searchsorted(cu[1:], idx, side="right")
    prev_seg = jnp.concatenate(
        [jnp.full((1,), -1, dtype=seg_id.dtype), seg_id[:-1]]
    )
    reset = (seg_id != prev_seg)
    a = jnp.where(reset[:, None], jnp.float32(0.0), a)

    def step(h, inp):
        a_t, dt_t, x_t, B_t, C_t = inp
        h = a_t[:, None, None] * h + (
            dt_t[:, None, None] * x_t[:, :, None] * B_t[:, None, :]
        )
        y_t = jnp.einsum("hpn,hn->hp", h, C_t)
        return h, y_t

    h0 = jnp.zeros((H, P, N), dtype=jnp.float32)
    _, y = jax.lax.scan(step, h0, (a, dt32, x32, B32, C32))
    return y


_reference_f32_jit = jax.jit(_reference_f32)


def check_correct(y: jax.Array, ref: jax.Array) -> tuple[bool, float, float]:
    y32 = y.astype(jnp.float32)
    diff = jnp.abs(y32 - ref)
    tol = ATOL + RTOL * jnp.abs(ref)
    worst = float(np.asarray((diff - tol).max()))
    max_abs = float(np.asarray(diff.max()))
    max_rel = float(np.asarray((diff / (jnp.abs(ref) + 1e-6)).max()))
    return worst <= 0.0, max_abs, max_rel


def time_fn(fn, args, warmup: int, iters: int) -> dict:
    """Return {median_us, mean_us, stdev_us, samples_us} after `warmup` discarded
    iterations and `iters` timed iterations."""
    for _ in range(warmup):
        y = fn(*args)
        y.block_until_ready()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn(*args)
        y.block_until_ready()
        samples.append((time.perf_counter() - t0) * 1e6)
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "stdev_us": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples_us": samples,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--case", type=Path, default=None,
        help="path to a case JSON (cases_mamba2_ssd/*.json schema). "
             "Defaults to a built-in b64 subset of the msl1024-sp0.95 workload.",
    )
    p.add_argument("--no-correctness", action="store_true",
                   help="skip fp32 reference check.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args(argv)

    case = (
        json.loads(args.case.read_text(encoding="utf-8"))
        if args.case is not None
        else DEFAULT_CASE
    )

    H = int(case["heads"])
    P = int(case["head_dim"])
    N = int(case["state_dim"])
    chunk = int(case["chunk_size"])
    if "seq_offsets" in case:
        cu_list = list(case["seq_offsets"])
        n_seqs = len(cu_list) - 1
        T = int(cu_list[-1])
        L_max = max(cu_list[i + 1] - cu_list[i] for i in range(n_seqs))
    else:
        lens = list(case["seq_lengths"])
        n_seqs = len(lens)
        T = int(sum(lens))
        L_max = int(max(lens))

    print(f"case: {case['case_id']}", file=sys.stderr)
    print(f"  H={H}  P={P}  N={N}  chunk={chunk}  dtype={case['dtype']}",
          file=sys.stderr)
    print(f"  n_seqs={n_seqs}  T={T}  L_max={L_max}", file=sys.stderr)
    print(f"  devices: {jax.devices()}", file=sys.stderr)

    inputs = materialize_case(case)

    # First call: triggers compilation + execution.
    print("compiling + running kernel ...", file=sys.stderr)
    y = ssd_candidate_jit(*inputs)
    y.block_until_ready()

    ok = True
    if not args.no_correctness:
        print("computing fp32 reference ...", file=sys.stderr)
        ref = _reference_f32_jit(*inputs)
        ref.block_until_ready()
        ok, max_abs, max_rel = check_correct(y, ref)
        verdict = "PASS" if ok else "FAIL"
        print(
            f"correctness: {verdict}  "
            f"max |diff|={max_abs:.3e}  max rel={max_rel:.3e}  "
            f"tol=atol({ATOL}) + rtol({RTOL})*|ref|",
            file=sys.stderr,
        )

    timing = time_fn(ssd_candidate_jit, inputs, args.warmup, args.iters)
    median_us = timing["median_us"]
    mean_us = timing["mean_us"]
    stdev_us = timing["stdev_us"]
    print(
        f"timing: warmup={args.warmup}  iters={args.iters}  L_max={L_max}\n"
        f"  median = {median_us:9.2f} us  ({median_us / 1000:.3f} ms)\n"
        f"  mean   = {mean_us:9.2f} us  ({mean_us / 1000:.3f} ms)\n"
        f"  stdev  = {stdev_us:9.2f} us",
        file=sys.stderr,
    )

    print(json.dumps({
        "case_id": case["case_id"],
        "T": T,
        "n_seqs": n_seqs,
        "L_max": L_max,
        "warmup": args.warmup,
        "iters": args.iters,
        "correct": bool(ok),
        "median_latency_us": round(median_us, 3),
        "mean_latency_us": round(mean_us, 3),
        "stdev_latency_us": round(stdev_us, 3),
    }, indent=2))

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

