# Remove the size_key clamp (kSizeKeyMax = 5) from per-length threshold indexing

## Status: Implemented.

## Goal

Remove the hardcoded 5-slot cap on per-length threshold arrays (`supp_thr` /
`conf_thr` / `lift_thr`). After the change, a rule of **any** total size can have
its **own** threshold slot; arrays may be any length; sizes beyond the array are
unconstrained. Motivation: per-size threshold ladders like `{2: 15, 6: 400,
99: 5000}` must be legal.

## Settled semantics (do not re-litigate)

1. **Direct indexing**: `size_key = total_size` (= |antecedent| + 1 = |Z|).
   Array index = `size_key - 1`. No clamp, no spillover, no "last entry covers
   all longer sizes".
2. **Beyond-array = no constraint**: index past the array end returns 0.0
   (already how `thr_at` behaves). A size not covered by the array is simply
   not threshold-checked.
3. This is a deliberate **behavior change** for rules with `total_size > 5`
   when a ≤5-slot array is passed: previously slot 5 governed them
   (`size_key = min(total_size, 5)`); now they are unconstrained unless the
   caller supplies a longer array. Rules of size ≤ 5 are completely unaffected
   (identical indexing), so every existing consumer result is preserved.
4. Wheel version stays **0.1.0** — downstream consumers vendor the wheel by
   exact filename (`fpm-0.1.0-cp311-cp311-linux_x86_64.whl`); the rebuild
   overwrites `dist/` in place. No `pyproject.toml` version bump.

## Current clamp locations (all of them — verified by grep)

| File | Line | What |
|---|---|---|
| `src/rules.h` | 14-18 | `Thresholds` comment + `static constexpr int kSizeKeyMax = 5;` |
| `src/rules.cpp` | 110 | `const int size_key = std::min<int>(static_cast<int>(zsize), Thresholds::kSizeKeyMax);` |
| `src/pymodule.cpp` | 109-110 | pybind docstring: "index = min(total_size, 5) - 1" |
| `src/pymodule.cpp` | 21-23 | `thr_vec` runtime cap: throws on arrays longer than 5 entries |
| `src/abi.h` | 36-37 | comment documents `size_key = min(total_size, 5) - 1` |
| `README.md` | 61-63 | documents `size_key = min(total_size, 5)` |
| `design_docs/DESIGN_FPM.md` | 139 | documents `size_key = min(total_size, K)` (default K=5) |
| `design_docs/DESIGN_FPM.md` | 170-172 | input sketch: `float64[K]` threshold arrays `(K = 5)` |
| `tests/test_rules.cpp` | 75 | brute-force oracle `oracle_rules()` encodes the same clamp |
| `tests/test_rules.cpp` | 359-403 | `TEST(Rules, SizeKeyClampAtFive)` — pins the old behavior |
| `python/test_fpm_python.py` | 38 | Python reference: `size_key = min(len(z), 5) - 1` |
| `python/test_fpm_python.py` | 167-169 | `test_invalid_inputs` pins the pybind length cap (6+ slots → ValueError) |

`thr_at` (`src/rules.cpp:14-17`) already bounds-checks and returns 0.0 past the
end — it needs **no** change; that is exactly the new beyond-array semantics.

The ABI (`src/abi.h`) takes arrays as pointer+length — **no ABI change**.

## Changes

### 1. Library code

- `src/rules.h`: delete `kSizeKeyMax`; rewrite the `Thresholds` comment to:
  per-length thresholds indexed by `size_key = total_size` (index `size_key - 1`);
  entries missing from the array mean 0.0 (no constraint).
- `src/rules.cpp:110`: replace the clamped line with
  `const int size_key = static_cast<int>(zsize);`
- `src/pymodule.cpp` docstring: "supp_thr/conf_thr/lift_thr: per-length
  thresholds, index = total_size - 1; entries beyond the array length mean no
  constraint".
- `src/pymodule.cpp` `thr_vec`: remove the runtime length cap (arrays longer
  than 5 entries threw `invalid_argument`); arrays may be any length.
- `src/abi.h`: update the threshold-indexing comment (no ABI change — arrays
  were always pointer + length).
- `README.md` and `design_docs/DESIGN_FPM.md`: update the documented indexing
  rule (and in DESIGN_FPM.md note this supersedes the original
  `min(total_size, K)` design point).

### 2. C++ tests (`tests/test_rules.cpp`)

- **Oracle fix (line 75)**: same one-line change as the engine —
  `const int size_key = static_cast<int>(z.size());`. The oracle stays an
  independent re-derivation from raw counts; it just implements the same
  addressing spec.
- **Rewrite `SizeKeyClampAtFive`** (renamed `DirectSizeKeyIndexing`) as a
  *discriminating-threshold* test that proves which slot is read. Extend the existing handcrafted dataset to 6-item
  itemsets: 850 tx of `{1,2,3,4,5,6}` + 150 tx of `{1,2,3,4,5,7}` (N = 1000;
  count({1,2,3,4,5}) = 1000, count({1,2,3,4,5,6}) = 850), `max_antecedents = 6`.
  Rule `{1,2,3,4,5} -> {6}` has conf = 0.85, |Z| = 6:
  - `conf_thr` slot5 = 0.0, slot6 = 0.9 → rule must be **dropped**
    (old clamped code would keep it, reading slot 5);
  - `conf_thr` slot5 = 0.9, slot6 = 0.0 → rule must be **kept**
    (old code would drop it).
  Both directions passing is only possible with direct indexing. Mirror the
  two cases for `supp_thr` (supp 0.85, thresholds 0.86) and add a lift case.
- **Beyond-array test**: same dataset, 5-slot array with slot5 = 0.9 → the
  size-6 rule must be **unconstrained** (kept); a size-5 rule with conf 0.85
  must still be dropped (slot 5 still applies to size 5).
- **Unit-level**: direct checks on the public `Thresholds` struct —
  7-slot vector: `conf_at(7)` returns element 6; 5-slot vector: `conf_at(7)`
  returns 0.0; `conf_at(0)`/`conf_at(1)` return 0.0 (size-1 rules don't exist).
- **`tests/test_abi.cpp`**: add a long-array pass-through case to
  `ThresholdArraysPassedThrough` (7-slot conf_thr array; assert behavior, not
  crash — the ABI is pointer+length).

### 3. Python reference (`python/test_fpm_python.py`)

- Line 38: `size_key = min(len(z), 5) - 1` → `size_key = len(z) - 1`. This file
  is the Python-side oracle used by the pytest layer; it must implement the
  same new spec.
- `test_invalid_inputs`: drop the "threshold array too long" case — it pinned
  the removed pybind length cap.
- Add `test_size_key_direct_indexing`: the discriminating size-6 cases and the
  beyond-array case, end-to-end through pybind (mirrors the C++ tests).
- `python/test_memory_stability.py` / `python/example.py`: no threshold-array
  usage; nothing to change.

## Build & verification

1. All three existing build trees must pass **fully green**:
   - `cmake --build build -j && cd build && ctest --output-on-failure` (Release)
   - same for `build-dbg` (Debug) and `build-san` (ASan+UBSan)
   Currently 37/37 in each; the count will grow with the new tests.
2. Rebuild the wheel in a **`python:3.11-bookworm`** container (GCC 12 —
   do not revert the existing `src/bitset.cpp` `hsum512()` workaround; plain
   `_mm512_reduce_add_epi64` trips a false-positive `-Wuninitialized` under
   GCC 12 + `-Werror`). Build with scikit-build-core, e.g.
   `pip wheel . --no-deps -w dist` (pyproject sets `build-dir = "build-wheel"`,
   `FPM_BUILD_TESTS=OFF`).
3. Native smoke test of the built wheel: install it into a scratch venv and
   mine a tiny CSR with a 7-slot `conf_thr` array and a planted size-6 rule —
   assert the size-6 slot governs (mirrors the C++ discriminating test,
   end-to-end through pybind). Downstream vendoring/installation into
   consuming applications is their own process, out of scope here.
4. Optional but cheap: `scripts/fpm_parity_check.py` +
   `scripts/run_spark_reference.py` exist for Spark/mlxtend parity on small
   data — re-run on a small dataset at flat thresholds (parity configs are
   unaffected by the clamp change; this guards against accidental collateral
   damage).

## Out of scope

- Any change to mining depth limits (`max_len`, `max_antecedents` stay
  caller-controlled parameters) or to `min_confidence` semantics.
- Version bump / PyPI publishing.
