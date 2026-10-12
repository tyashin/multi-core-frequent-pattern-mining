"""Timing comparison: fpm vs PySpark FPGrowth on the same dataset.

Spark costs are split into: session startup (reported, excluded from compute),
Python->Spark ingest (createDataFrame), fit (frequent itemsets), and rule
materialization (associationRules is lazy in Spark ML — forced by an action).
fpm numbers are single-call: mine_itemsets (build + mine) and mine (build +
mine + rules), no witnesses.

Usage: python bench/bench_vs_spark.py DATASET.dat.gz MIN_SUPPORT MIN_CONFIDENCE [MAX_LEN] [DRIVER_MEM_GB]
"""

import gzip
import math
import sys
import time

import numpy as np

import fpm


def load_fimi(path):
    offsets = [0]
    keys = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            keys.extend(int(tok) for tok in line.split())
            offsets.append(len(keys))
    return np.array(offsets, dtype=np.uint64), np.array(keys, dtype=np.uint64)


def main():
    dataset, min_support, min_confidence = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    max_len = (int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].lower() != "none"
               else None)
    driver_mem = f"{sys.argv[5]}g" if len(sys.argv) > 5 else "4g"

    offsets, keys = load_fimi(dataset)
    n = len(offsets) - 1
    minsup_abs = math.ceil(min_support * n)
    max_len_eff = max_len if max_len is not None else len(np.unique(keys))
    max_ant = max_len_eff - 1
    print(f"{dataset}: N={n}, minsup={min_support} -> {minsup_abs}, "
          f"min_conf={min_confidence}, max_len={max_len_eff}")

    # --- fpm ---
    t0 = time.perf_counter()
    fi = fpm.mine_itemsets(offsets, keys, minsup_abs, max_len_eff, 0)
    t_fpm_sets = time.perf_counter() - t0
    n_sets = len(fi["counts"])

    t0 = time.perf_counter()
    rules = fpm.mine(offsets, keys, min_support_abs=minsup_abs, max_len=max_len_eff,
                     max_antecedents=max_ant, min_confidence=min_confidence,
                     threads=0, k_witnesses=0)
    t_fpm_rules = time.perf_counter() - t0
    n_rules = len(rules["consequent"])
    print(f"fpm:    itemsets {n_sets} in {t_fpm_sets * 1000:9.1f} ms | "
          f"rules {n_rules} (full pipeline) in {t_fpm_rules * 1000:9.1f} ms")

    # --- Spark ---
    from pyspark.sql import SparkSession
    from pyspark.ml.fpm import FPGrowth

    t0 = time.perf_counter()
    spark = (SparkSession.builder.master("local[*]").appName("fpm-spark-bench")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.driver.memory", driver_mem).getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    t_start = time.perf_counter() - t0

    t0 = time.perf_counter()
    opener = gzip.open if dataset.endswith(".gz") else open
    rows = []
    with opener(dataset, "rt") as fh:
        for line in fh:
            rows.append((list(dict.fromkeys(line.split())),))  # Spark needs unique items
    df = spark.createDataFrame(rows, ["items"])
    df.count()  # force materialization of the input DataFrame
    t_ingest = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = FPGrowth(minSupport=min_support, minConfidence=min_confidence,
                     itemsCol="items").fit(df)
    n_sets_spark = model.freqItemsets.count()  # action; also validates fit
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_rules_spark = model.associationRules.count()  # forces lazy rule generation
    t_rules = time.perf_counter() - t0

    print(f"spark:  startup {t_start:5.1f}s | ingest {t_ingest:5.1f}s | "
          f"itemsets {n_sets_spark} in {t_fit:8.1f}s | rules {n_rules_spark} in {t_rules:8.1f}s")
    spark_compute = t_fit + t_rules
    print(f"speedup (compute only, startup+ingest excluded): "
          f"itemsets {t_fit / t_fpm_sets:7.1f}x | full pipeline {spark_compute / t_fpm_rules:7.1f}x")
    if n_sets_spark != n_sets or n_rules_spark != n_rules:
        print(f"WARNING: count mismatch (sets {n_sets} vs {n_sets_spark}, "
              f"rules {n_rules} vs {n_rules_spark}) — run scripts/fpm_parity_check.py")
    spark.stop()


if __name__ == "__main__":
    main()
