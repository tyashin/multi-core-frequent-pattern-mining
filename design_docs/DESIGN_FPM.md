# FPM — Multi-Core Frequent-Pattern Mining Library (C++)

## Status: Draft — standalone library, Phase 1. Supersedes an earlier Mojo-based design draft (decision history in git).

## Goal

A fast, embeddable, **multi-core frequent-itemset and association-rule mining library** with a Python interface, usable from any host application.

The niche: single-machine frequent-pattern mining is split between slow single-core libraries (mlxtend, Borgelt's C) and cluster frameworks (Spark PFP) whose overhead dominates at single-machine scale. Nothing mature occupies the middle: one process, all cores, native speed, called from Python. This library occupies it.

**Boundary rules (what the library is not):**

- Items are **opaque `uint64` ids** — callers own their mapping; the library never sees application semantics.
- **No database access, no application logic, no host-application names or imports** anywhere in the project.
- Configuration (thresholds, depth caps, witness counts) is passed per call; nothing is read from files or environment beyond documented knobs.

## Decision Ledger: Why C++

Alternatives evaluated (full trail in the appendix):

- **C++ chosen** for: toolchain maturity (distro gcc/clang, nothing to pin); every identified optimization available on day one (SIMD intrinsics, non-temporal stores, OpenMP, terminal-level store-skip); a small, fully self-owned kernel (no code borrowing, no license questions); Highway available if cross-arch SIMD is ever needed.
- **Residual risk, accepted**: memory-safety footguns of raw C++. Mitigations: small codebase behind one ABI boundary, ASan/UBSan on the full test suite in CI, property tests against an independent oracle.

---

## 1. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | C++17 (GCC 12+) | nothing needed beyond C++17 |
| Build | CMake → `libfpm.so` + Python module `fpm` | multi-stage Docker builds: compile in builder, copy artifacts into slim runtime |
| Packaging | scikit-build-core → wheel (`pip install ./fpm`) | consumers install it as a normal Python dependency |
| Parallelism | OpenMP (`#pragma omp parallel for if(...)`) | the `if` clause natively implements the work-threshold pattern |
| SIMD | AVX2 + AVX-512 intrinsics, runtime CPU dispatch, scalar fallback | one binary, three ISA variants via GCC `target` attributes; Highway as a future option for ARM |
| Python interface | **pybind11** (header-only, build-time-only dependency) | real extension module; raw `extern "C"` + ctypes kept as zero-dependency fallback |
| C++ tests | GoogleTest (CMake FetchContent) | ASan/UBSan configurations in CI |
| Integration tests | pytest | plus PySpark/mlxtend as **dev-only** validation oracles |
| CI hygiene | `-Wall -Wextra -Werror`, sanitizers on tests | small codebase, held to a high bar |

---

## 2. Reference Project: `mojo-mlxtend`

[`github.com/lee101/mojo-mlxtend`](https://github.com/lee101/mojo-mlxtend) (MIT) is a Mojo port of mlxtend's frequent-pattern module with compute in a compiled Mojo shared library.

**What it proved — the reason this design exists:**

- A **vertical bitset (Eclat-style) engine** behind an FP-growth-compatible API returns identical frequent itemsets while being SIMD-native: transactions packed into item-major `uint64` bitsets, support = AND + popcount.
- Measured benchmarks (author's, parity-asserted against mlxtend 0.25.0, single core): **fpgrowth 29.5x faster than mlxtend** (26.8 ms vs 790 ms on 40k×32), fpmax 20.7x, apriori 11.9x.
- Its Apriori kernel demonstrated the **threshold-gated parallelism** pattern: parallelize a batch loop only when `work ≥ threshold`, else stay serial.
- Its README's GPU analysis: these kernels are **bandwidth-bound** (< 2 ops/byte moved); a GPU loses to the CPU once host/device copies are counted.

**What we take:** the architecture, the kernel design, the parallelization pattern, the benchmark reference points.

**What we do NOT take:** any code (the C++ kernel is a few hundred lines of standard techniques — nothing worth borrowing), its Python recursion driver (bottleneck at pattern-explosion scale — our driver is native C++), its ABI shape, its mlxtend compatibility shims.

**Why not simply use it:** fpgrowth path single-threaded, Python driver in the hot path, nightly toolchain pins. It is the prototype that validated the design; this library is the production implementation of the same design.

---

## 3. The Algorithm: Vertical Bitset Mining (Eclat-Style)

The engine does **not** implement the classic FP-tree. It uses a vertical representation — identical outputs, different internals.

### Representation

Transpose the database: instead of "transaction → items", store "item → transactions".

```
B[item] = uint64_t[ words ],  words = ceil(N / 64)
bit i of B[item] = 1  ⇔  item occurs in transaction i
```

Support of any itemset `X`: `supp(X) = popcount(B[x1] & B[x2] & … & B[xk])`.

### Search (depth-capped DFS over conditional pattern bases)

```
mine(parent_bits, candidates, depth):
    for item in candidates:                       # batched, parallel over items
        child_bits = parent_bits & B[item]        # skipped at terminal depth
        c = popcount(child_bits)
        if c >= min_support_abs:
            emit(itemset = prefix ∪ {item}, count = c)
            if depth + 1 < max_len:
                mine(child_bits, items after `item`, depth + 1)
```

- **Depth cap (early filtering by antecedent length).** When mining rules with a single consequent and an antecedent-size cap `A`, no itemset longer than `A + 1` can produce a valid rule — so `max_len = A + 1` loses nothing and prunes most of the low-support pattern space. Note the two-stage structure: the antecedent cap `A` also appears as a *late* filter at rule emission (Section 4), but the decisive win is applying it **early** — as a hard bound on the search itself, so long itemsets are never even generated. This cap (absent from Spark MLlib, which can only filter late) is the single largest speedup at low support.
- Candidates per node: items **after** the current item in a fixed global order (canonical Eclat ordering — no duplicate enumeration, no Apriori-style candidate generation).
- **Output equivalence**: the frequent-itemset set `{X : supp(X) ≥ minsup}` is defined by data and threshold; FP-growth, Eclat, and this engine enumerate the same set with the same integer counts — which is what makes strict output validation meaningful.

### Parallelism

Within each conditional step, children are computed as a batch: each child writes to a **disjoint** output slot, reads shared read-only inputs, and costs a uniform `words`-length pass — embarrassingly parallel, no synchronization:

```cpp
#pragma omp parallel for schedule(dynamic, 64) \
        if((int64_t)item_count * words >= PARALLEL_THRESHOLD)
    for (int64_t i = 0; i < item_count; ++i)
        child[i] = intersect_count(parent, B[item[i]], out[i], words);
```

Thread count is **detected at runtime, never hardcoded** — the core count is not known in advance (dev machines, servers, container limits). Default `threads = 0` uses the OpenMP runtime's processor count and honors `OMP_NUM_THREADS`; an explicit value overrides. Capping near physical cores stays the *policy* (bandwidth-bound kernel; SMT adds nothing), but the number is discovered, not assumed. Container caveat: cpuset limits are respected via `sched_getaffinity`, but CPU *quota* (cgroup cfs) is not — under quota, set `OMP_NUM_THREADS` (or pass `threads`).

### Kernel-level optimizations

1. **Terminal-level store-skip.** At `depth == max_len - 1`, children are only *counted*, never recursed — no child bitset is materialized. The terminal level dominates node count at low support; skipping its store stream cuts that level's memory traffic by a third.
2. **Non-temporal stores.** Child bitsets at non-terminal levels are write-once, read-later — streamed (`_mm256_stream_si256` + `_mm_sfence()`), avoiding read-for-ownership traffic (~25–30% less total memory traffic on store-heavy levels). Buffers 64-byte aligned.
3. **Runtime ISA dispatch.** AVX-512 (`VPOPCNTQ`), AVX2 (Muła `pshufb` nibble-LUT popcount), scalar (`__builtin_popcountll`) — one binary, selected once at load via `__builtin_cpu_supports`.
4. **Threshold-gated parallelism** (pattern credited to mojo-mlxtend's Apriori kernel): small batches stay serial.

### Memory

- Bitsets: `N × M_freq / 8` bytes (1M transactions × 50k items ≈ 6 GB).
- Per-thread child buffers: `batch × words × 8 B × threads`, bounded and reused.
- Witness pass: streamed, no retention of union bitsets.

### Bandwidth ceiling

The kernel performs ~0.1 arithmetic ops per byte moved — **memory-bandwidth-bound**. One core streams 10–20 GB/s; a server socket saturates around 60–150 GB/s, so parallel scaling flattens at ~8–15× regardless of core count. Consequences: ~12 physical cores is the sweet spot; a GPU cannot help (PCIe is often slower than the CPU's own bus); a cluster adds capacity, not speed.

---

## 4. Rule Generation and Reference Semantics

### Rules

For each frequent itemset `Z` with `2 ≤ |Z| ≤ max_len`, emit rules `X → {c}` (single consequent), computed from **integer counts** collected during mining (fractions derived only at the end — no rounding drift):

- `support(Z) = count(Z) / N`, `confidence = count(Z) / count(X)`, `lift = confidence / (count({c}) / N)`

### Threshold filtering (generic)

Per-call parameters, applied to every rule:

- global `min_confidence` floor (applied at generation, Spark-compatible semantics);
- antecedent-size cap `A`;
- optional **per-length thresholds**: arrays indexed by `size_key = total_size` (entry index `size_key - 1`) for support / confidence / lift; sizes beyond the array length default to 0.0 (no constraint). (Supersedes the original `min(total_size, K)` clamp with `K = 5` — see `design_docs/REMOVE_SIZE_KEY_CLAMP.md`.)

**Parameter policy (guidance for callers).** To make antecedent length vary with support — shorter rules at low support, longer at high support — express it through the **per-length support thresholds** (set `supp_thr` high for large sizes, so long rules survive only on strong evidence), not by coupling `max_len` to `min_support_abs`. The per-length mechanism is smooth (no discontinuity when support crosses a boundary) and keeps the two knobs independent. A hard "if support < X then A = 3" rule, if wanted operationally, belongs in the consumer's configuration as a default policy table — never inside the library.

### Spark-compatibility modes (for validation)

Two Spark MLlib behaviors are replicated so output can be validated against a PySpark reference on the same dataset:

1. Support threshold conversion: `ceil(fraction × N)` semantics (the float round-trip can make this `abs + 1` for some N).
2. `minConfidence` applied at rule **generation** time, i.e. effective confidence threshold `max(min_confidence, conf_thr[size_key])`.

### Witnesses and counts

- `n_baskets` per rule = the union itemset's integer count, already known from mining — free.
- Union bitsets are **not retained** during mining (retention would be unbounded). Witnesses are computed in a bounded second pass over surviving rules only: re-intersect the rule's ≤ `max_len` item bitsets word-by-word and **stop after `k` set bits** (trailing-zero-count walk). Bounded work per rule, parallelizable over rules; the library returns witness **transaction indices** — callers map indices to their own identifiers.

---

## 5. Python Interface

Normal extension module. One call in, one call out; all allocations owned by the C++ side and returned as numpy arrays — no sizing protocol, no address-as-int passing, no error-code channel; C++ exceptions surface as Python exceptions.

### Input

```python
result = fpm.mine(
    offsets,          # uint64[N+1]   CSR row pointers into keys
    keys,             # uint64[M]     item ids (caller-defined packing)
    min_support_abs,  # int           absolute support, in transactions
    max_len,          # int           itemset depth cap (e.g. A + 1)
    max_antecedents,  # int           rule antecedent-size cap A
    supp_thr,         # float64[]     per-length thresholds (index = total_size - 1)
    conf_thr,         # float64[]
    lift_thr,         # float64[]
    min_confidence,   # float         global generation-time floor
    threads,          # int           0 = auto-detect (default); explicit value overrides
    k_witnesses,      # int           witnesses per rule (e.g. 5)
)
```

Notes: duplicate items within a transaction should be removed during CSR construction. Fractional support, if a caller prefers it, is resolved to an absolute count by the caller before the call.

### Output (dict of numpy arrays, CSR-shaped)

```python
result["ant_offsets"]     # uint64[R+1]   antecedent CSR pointers
result["ant_keys"]        # uint64[...]   antecedent item ids
result["consequent"]      # uint64[R]     consequent item id
result["support"]         # float64[R]
result["confidence"]      # float64[R]
result["lift"]            # float64[R]
result["n_baskets"]       # int64[R]      exact union counts
result["witnesses"]       # int32[R, k]   transaction indices, -1 padded
```

### C++ signature sketch

```cpp
MineResult mine(const py::array_t<uint64_t>& offsets,
                const py::array_t<uint64_t>& keys,
                int64_t min_support_abs, int max_len, int max_antecedents,
                Thresholds thr, double min_confidence,
                int threads, int k_witnesses)
{
    py::gil_scoped_release release;   // mining runs without the GIL
    ...
}
```

GIL released during mining; OpenMP threads are native. Buffers read via `py::buffer_info` — zero-copy, no Python objects touched after the boundary.

### Fallback: raw C ABI

The core library also exposes a minimal `extern "C"` entry point (buffers as pointers, output into caller-allocated storage) for non-Python consumers or ctypes purists. The core knows nothing about Python; both wrappers are thin.

---

## 6. Project Layout (standalone git repository)

```
fpm/                              # standalone project — its own git repo
├── CMakeLists.txt                # targets: fpm_core (static), fpm (pybind11 module), fpm_tests
├── pyproject.toml                # scikit-build-core → wheel
├── README.md                     # API + build docs
├── src/
│   ├── bitset.h / bitset.cpp     # layout, allocation, scalar/AVX2/AVX-512 intersect+popcount, NT stores
│   ├── mine.h / mine.cpp         # vertical build + capped-depth DFS driver (OpenMP, runtime thread detection)
│   ├── rules.h / rules.cpp       # rule expansion, metrics, generic per-length threshold filter
│   ├── witness.h / witness.cpp   # bounded witness extraction pass
│   ├── abi.h / abi.cpp           # extern "C" fallback ABI
│   └── pymodule.cpp              # pybind11 wrapper (thin)
├── tests/                        # GoogleTest: kernels, driver, edge cases (ASan/UBSan configs)
├── python/                       # pytest integration + example
├── bench/                        # vs mlxtend oracle on FIMI datasets (kosarak, retail, accidents)
└── scripts/
    ├── run_spark_reference.py    # PySpark reference runner (dev-only validation)
    └── fpm_parity_check.py       # output diff/validation harness
```

---

## 7. Verification

**Unit-test checklist (GoogleTest, run under ASan+UBSan in CI):** key packing/unpacking round-trip; popcount vs. naive counting on random buffers; intersect semantics (disjoint / identical / empty inputs); CSR construction with duplicate items per transaction, empty transactions, single-item transactions; support boundary ±1 around `min_support_abs`; `max_len` cutoff exactness (nothing longer than the cap, nothing missing below it); threshold-filter matrix (every `size_key`, every metric, defaults for missing keys); terminal-level store-skip correctness (counts identical with store on/off, via debug flag); witness early-exit (exactly `k`, validity, `-1` padding); determinism `threads=1` vs `threads=N` (identical outputs regardless of scheduling); ISA-dispatch equivalence (scalar / AVX2 / AVX-512 forced via env override — bitwise-identical results); ABI hygiene (repeated calls, exception propagation to Python, no leaks across the boundary).

**Independent oracles:**

- **mlxtend**: random synthetic datasets cross-checked against it (slow but fully independent); FIMI benchmark datasets (kosarak, retail, accidents) with known itemset counts.
- **PySpark**: `scripts/run_spark_reference.py` — a small PySpark FPGrowth runner that reads a dataset and writes reference output; `scripts/fpm_parity_check.py` diffs library output against it on **the same dataset**: rule-key sets **0 missing / 0 extra**; metrics within `1e-9` relative; `n_baskets` exact; witness **validity** (each witness transaction verifiably contains all rule items — witness *identity* is unstable across implementations, validity is not); boundary audit around the support threshold; input checksum logged. PySpark + JRE are **dev-only dependencies** of the test environment; the library and its wheel never depend on them.

**Downstream/production integration validation** (shadow tables, live-traffic comparison, rollout gates) belongs to the consuming application's repository and process — out of scope for this library.

---

## 8. Implementation Steps & Effort

| Step | Content | Estimate |
|---|---|---|
| 1 | CMake skeleton, bitset kernels (3 ISA variants) + unit tests | ~1 day |
| 2 | Native driver: index, vertical build, capped DFS with OpenMP, runtime thread detection | ~1–2 days |
| 3 | Rules + thresholds + witness pass | ~1 day |
| 4 | pybind11 wrapper, wheel packaging, README | ~1 day |
| 5 | mlxtend oracle on FIMI datasets + benchmarks | ~0.5–1 day |
| 6 | PySpark reference runner + parity harness | 1–2 days |

Total new C++ ≈ 600–900 lines. The parity/verification work is deliberately the larger half of the effort.

---

## Appendix

### Rejected alternatives

| Option | Why rejected |
|---|---|
| Spark PFP (as embeddable approach) | JVM/driver overhead dominates single-machine workloads; no itemset-depth cap exists |
| Borgelt C / pyfim | Single-core; parallelizing it is a rewrite of shared-state 2005-era C; commercial license unverified |
| mojo-mlxtend as-is | fpgrowth path single-threaded; Python driver in the hot path; nightly toolchain — used as design prototype instead |
| Mojo | Young toolchain/nightly pins; NT-store exposure unverified; smaller maintenance pool |
| Rust | Equal performance port target; declined for team reasons |
| Go | ~20–30% DRAM-wall efficiency price (no SIMD, no NT stores); easiest team adoption |
| Go + assembly | SIMD at the cost of the least maintainable, non-portable artifact |
| GPU | Bandwidth-bound kernel; PCIe transfer tax exceeds any gain |
| Research C++ (PFP-Growth, PrefixFPM, freqmine) | 2017–2019 academic code, unmaintained, correctness unverified |

### The output-bound floor

At very low support, the surviving-rule count and the consumer's write volume — not the engine — set the runtime floor. If runs stay slow after migration, the lever is threshold policy (per-length thresholds), not compute.

### Cluster escape hatch

Single-node shared-memory by design. If data ever outgrows one machine's RAM: replicate the compact bitset file to each node once, partition work by top-level item ranges across nodes, merge results. No transaction shuffle needed. Out of scope.

### References

- Han, Pei, Yin. *Mining Frequent Patterns without Candidate Generation.* SIGMOD 2000.
- Zaki. *Scalable Algorithms for Association Mining* (Eclat). IEEE TKDE, 2000.
- Borgelt. *An Implementation of the FP-growth Algorithm.* OSDM 2005. https://borgelt.net/fpgrowth.html
- mojo-mlxtend (MIT) — design prototype and benchmark reference. https://github.com/lee101/mojo-mlxtend
- Muła, Kurz, Lemire. *Faster Population Counts Using AVX2 Instructions.* 2016.
- Spark MLlib FPGrowth (reference semantics for validation). https://spark.apache.org/docs/latest/ml-frequent-pattern-mining.html
