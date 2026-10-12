# fpm — multi-core frequent-pattern mining

A fast, embeddable **frequent-itemset and association-rule mining library**: one
process, all cores, native (C++17) speed, called from Python. Vertical bitset
(Eclat-style) engine behind a single `fpm.mine()` call.

Items are **opaque `uint64` ids** — callers own their mapping; the library never
sees application semantics. No database access, no configuration files.

## Install

```bash
pip install .          # builds the native extension via scikit-build-core
```

Runtime dependency: the OpenMP runtime (`libgomp`, present wherever gcc is).
Build-time only: CMake, pybind11, scikit-build-core (auto-fetched by pip).

## Usage

```python
import numpy as np
import fpm

offsets = np.array([0, 3, 5, 9], dtype=np.uint64)      # CSR row pointers
keys = np.array([10, 20, 30, 10, 20, 10, 20, 30, 40], dtype=np.uint64)

result = fpm.mine(
    offsets, keys,
    min_support_abs=2,      # absolute support, in transactions
    max_len=3,              # itemset depth cap (e.g. max_antecedents + 1)
    max_antecedents=2,      # rule antecedent-size cap A
    supp_thr=np.zeros(0),   # per-length thresholds (optional)
    conf_thr=np.zeros(0),
    lift_thr=np.zeros(0),
    min_confidence=0.5,     # global generation-time floor
    threads=0,              # 0 = auto-detect (honors OMP_NUM_THREADS)
    k_witnesses=5,          # witness transactions per rule (0 = skip)
)
```

Output is a dict of numpy arrays, CSR-shaped:

| key | dtype | shape | meaning |
|---|---|---|---|
| `ant_offsets` | uint64 | `(R+1,)` | antecedent CSR pointers |
| `ant_keys` | uint64 | `(sum ant lens,)` | antecedent item ids |
| `consequent` | uint64 | `(R,)` | consequent item id |
| `support` | float64 | `(R,)` | `count(Z) / N` |
| `confidence` | float64 | `(R,)` | `count(Z) / count(X)` |
| `lift` | float64 | `(R,)` | `confidence / (count({c}) / N)` |
| `n_baskets` | int64 | `(R,)` | exact union counts |
| `witnesses` | int32 | `(R, k)` | transaction indices, `-1` padded |

Notes:

- Remove duplicate items within a transaction during CSR construction
  (results stay correct if you don't; it's just wasted work).
- Fractional support: resolve to an absolute count before the call
  (Spark-compatible: `ceil(fraction * N)`).
- **Per-length thresholds**: arrays indexed by `size_key = min(total_size, 5)`,
  entry index `size_key - 1`; missing entries mean no constraint. Effective
  confidence floor per rule is `max(min_confidence, conf_thr[size_key])`
  (Spark-compatible generation-time semantics). To make antecedent length vary
  with support, express it through `supp_thr` (high thresholds for large
  sizes), not by coupling `max_len` to `min_support_abs`.
- Witnesses are transaction **indices**; map them to your own identifiers.
  Witness *identity* is implementation-specific; witness *validity* (each
  witness contains all rule items) is guaranteed.

## Threads and environment knobs

- `threads=0` uses the OpenMP runtime's processor count and honors
  `OMP_NUM_THREADS`; an explicit value overrides. Cap near physical cores —
  the kernel is memory-bandwidth-bound, SMT adds nothing.
- Containers: cpuset limits are respected; CPU *quota* (cgroup cfs) is not —
  under quota, set `OMP_NUM_THREADS` (or pass `threads`).
- `FPM_FORCE_ISA=scalar|avx2|avx512` forces the kernel ISA (testing/debugging;
  unsupported requests are clamped).

## C ABI

`libfpm.so` exposes a minimal `extern "C"` entry point (`src/abi.h`) for
non-Python consumers: one call in, library-owned `FpmRules*` out, freed with
`fpm_free_rules`. Errors: `nullptr` return + `fpm_last_error()`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j        # builds libfpm.so, fpm_tests
```

## Development

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/fpm_tests                                        # GoogleTest suite
cmake -S . -B build-san -DFPM_SANITIZE="address,undefined"  # sanitizer build
cmake --build build-san -j && ./build-san/fpm_tests
python -m pytest python/                                 # Python integration
```

`python/example.py` is a minimal end-to-end example. `scripts/` holds the
dev-only validation harness (PySpark reference + parity check); PySpark and a
JRE are dev-only dependencies — the library and its wheel never need them.

## Design

See `design_docs/DESIGN_FPM.md`. The engine transposes the database into
item-major `uint64` bitsets (`support = AND + popcount`), runs a depth-capped
DFS with threshold-gated OpenMP parallelism, terminal-level store-skip,
non-temporal stores, and runtime ISA dispatch (scalar / AVX2 / AVX-512). The
kernel is memory-bandwidth-bound: scaling flattens at ~8–15×, ~12 physical
cores is the sweet spot.
