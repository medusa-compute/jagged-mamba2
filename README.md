# jagged-mamba2

Single Pallas kernel for jagged Mamba2 SSD prefill on TPU v6e.

Implementation: [`mamba2-candidate-v6.py`](mamba2-candidate-v6.py)

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

```python
import jax, jax.numpy as jnp, importlib.util, sys

spec = importlib.util.spec_from_file_location("v6", "mamba2-candidate-v6.py")
v6 = importlib.util.module_from_spec(spec); sys.modules["v6"] = v6
spec.loader.exec_module(v6)

seqlens = jnp.array([200, 500, 324], jnp.int32)
cu_seqlens = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(seqlens)])
T, H, P, N = int(cu_seqlens[-1]), 128, 64, 128

k = jax.random.split(jax.random.PRNGKey(0), 5)
x  = jax.random.normal(k[0], (T, H, P), jnp.bfloat16)
dt = jax.nn.softplus(jax.random.normal(k[1], (T, H), jnp.bfloat16))
A_log = -jnp.exp(jax.random.normal(k[2], (H,), jnp.float32))
B  = jax.random.normal(k[3], (T, H, N), jnp.bfloat16)
C  = jax.random.normal(k[4], (T, H, N), jnp.bfloat16)

y = v6.ssd_candidate_jit(x, dt, A_log, B, C, cu_seqlens)
y.block_until_ready()
```

## Shape constraints

- `H == 128` (one program handles all heads).
- `CHUNK == 256`, `I_TILE == 64`.
- `P`, `N` adjustable, subject to VMEM budget: `h_init` scratch is `(128, P, N) f32` plus emit_pipeline double-buffering.
