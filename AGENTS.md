# AGENTS.md — fpm (frequent-pattern mining library)

C++17 vertical-bitset engine (Eclat-style) + pybind11 module, built with CMake /
scikit-build-core. Single-consequent association rules with per-length thresholds.

## Build & test

Three existing CMake trees, all must stay green (`ctest --output-on-failure`):

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure      # Release
cmake --build build-dbg -j && ctest --test-dir build-dbg --output-on-failure  # Debug
cmake --build build-san -j && ctest --test-dir build-san --output-on-failure  # ASan+UBSan
```

Python integration tests run from `.venv`:

```bash
.venv/bin/python -m pytest python/ -q
```

## Python environment: `.venv` has no pip

The venv is Python 3.12 with pytest/numpy/pyspark/mlxtend but **no pip module**.
Use `uv` (at `~/.local/bin/uv`). Reinstall the extension after C++ changes with:

```bash
uv pip install --python .venv/bin/python . --no-deps --force-reinstall -C build-dir=build-host
```

The `-C build-dir=build-host` matters: `pyproject.toml` pins `build-dir = "build-wheel"`
for the official wheel — don't clobber it with host builds.

## Wheel builds: bookworm container only

The shipped wheel (`fpm-0.1.0-cp311-cp311-linux_x86_64.whl`, version pinned —
no bumps) must be built in a `python:3.11-bookworm` container (GCC 12). Do not
revert the `hsum512()` workaround in `src/bitset.cpp`: plain
`_mm512_reduce_add_epi64` trips a `-Wuninitialized` false positive under
GCC 12 + `-Werror`. `dist/` is gitignored; downstream consumers vendor the
wheel by exact filename.

## Benchmarks & parity (dev-only deps: mlxtend, PySpark + JRE)

Datasets live in `bench/data/` (gitignored: retail/kosarak/accidents `.dat.gz`).

```bash
.venv/bin/python bench/bench_vs_mlxtend.py --all
.venv/bin/python bench/bench_vs_spark.py bench/data/<ds>.dat.gz <minsup> <conf> [max_len] [driver_mem_gb]
```

accidents@0.2 under Spark needs a 12 GB driver heap (last arg `none 12`).
README benchmark tables are single-run numbers from this machine (i5-9400F) —
re-run before publishing changed numbers. Parity harness:
`scripts/run_spark_reference.py` then `scripts/fpm_parity_check.py`.

## Conventions

- `design_docs/` holds design specs; keep them in sync with behavior changes.
- Threshold semantics, Spark-compat modes, and engine design: `design_docs/DESIGN_FPM.md`.
- `-Wall -Wextra -Werror` is on; treat warnings as errors locally too.
