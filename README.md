# jagged-mamba2

Single Pallas kernel for jagged Mamba2 SSD prefill on TPU v6e.

Implementation: [`candidate_v6.py`](candidate_v6.py). Standalone driver with correctness check and timing: [`main.py`](main.py).

## Environment

| | |
|---|---|
| Hardware | TPU v6e (1x1), 32 GB HBM |
| Python | 3.11 |
| JAX | 0.10.0 |
| libtpu | 0.0.40 |

```bash
pip install "jax[tpu]==0.10.0" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -c "import jax; print(jax.devices())"   # expect TpuDevice
```

Pallas TPU is tightly coupled to JAX/libtpu versions — wrong versions either fail at lowering or silently miscompute. Don't upgrade casually.

## API

`ssd_candidate(x, dt, A_log, B, C, cu_seqlens) -> y`

| Tensor | Shape | dtype |
|---|---|---|
| `x` | `(T, H, P)` | bf16 |
| `dt` | `(T, H)` | bf16 |
| `A_log` | `(H,)` | f32 |
| `B`, `C` | `(T, H, N)` | bf16 |
| `cu_seqlens` | `(num_seqs+1,)` | i32 |
| `y` (out) | `(T, H, P)` | bf16 |

Hard-coded: `H=128, P=64, N=128, CHUNK=256, I_TILE=64`. `T` need not be a multiple of `CHUNK` — the wrapper pads and slices.

## Run

The bundled driver runs a deterministic 64-sequence workload (T=19120, the first 64 sequences of `mamba2_ssd-msl1024-b512-sp0.95-a2-h128p64n128-bf16`, sized to fit one v6e chip), checks against an inline fp32 reference, and prints median latency:

```bash
python3 main.py                              # default workload + correctness + timing
python3 main.py --case path/to/case.json     # custom case (cases_mamba2_ssd/*.json schema)
python3 main.py --no-correctness             # skip the fp32 ref (~5x slower than the kernel)
python3 main.py --warmup 5 --iters 20        # tweak timing
```

Or call the kernel directly:

```python
import jax, jax.numpy as jnp
from candidate_v6 import ssd_candidate_jit

seqlens = jnp.array([200, 500, 324], jnp.int32)
cu_seqlens = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(seqlens)])
T, H, P, N = int(cu_seqlens[-1]), 128, 64, 128

k = jax.random.split(jax.random.PRNGKey(0), 5)
x  = jax.random.normal(k[0], (T, H, P), jnp.bfloat16)
dt = jax.nn.softplus(jax.random.normal(k[1], (T, H), jnp.bfloat16))
A_log = -jnp.exp(jax.random.normal(k[2], (H,), jnp.float32))
B  = jax.random.normal(k[3], (T, H, N), jnp.bfloat16)
C  = jax.random.normal(k[4], (T, H, N), jnp.bfloat16)

y = ssd_candidate_jit(x, dt, A_log, B, C, cu_seqlens)
y.block_until_ready()
```

## Shape constraints

- `H == 128` (one program handles all heads).
- `CHUNK == 256`, `I_TILE == 64`.
- `P`, `N` adjustable, subject to VMEM budget: `h_init` scratch is `(128, P, N) f32` plus emit_pipeline double-buffering.
