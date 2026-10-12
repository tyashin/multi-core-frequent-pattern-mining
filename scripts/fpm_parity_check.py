"""Parity check: fpm library output vs a PySpark FPGrowth reference, same dataset.

Checks (per design doc section 7):
  - rule-key sets: 0 missing / 0 extra
  - support / confidence / lift within 1e-9 relative
  - n_baskets exact (from the Spark freqItemsets freq column)
  - witness validity: each witness transaction verifiably contains all rule items
  - boundary audit: itemsets with count in {minsup-1, minsup, minsup+1} are
    included/excluded exactly per the threshold
  - input checksum: dataset sha256 must match the reference's meta.json

Usage:
  python scripts/fpm_parity_check.py REF_DIR [--max-antecedents A] [--k-witnesses K]
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import sys

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


def rel_close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir")
    ap.add_argument("--max-antecedents", type=int, default=10**9)
    ap.add_argument("--k-witnesses", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.ref_dir, "meta.json")) as fh:
        meta = json.load(fh)
    n = meta["n_transactions"]
    min_support = meta["min_support"]
    min_confidence = meta["min_confidence"]
    dataset = meta["dataset"]

    h = hashlib.sha256()
    with open(dataset, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == meta["dataset_sha256"], "dataset checksum mismatch"
    print(f"input checksum OK ({h.hexdigest()[:16]}...)")

    offsets, keys = load_fimi(dataset)
    assert len(offsets) - 1 == n, "transaction count mismatch vs meta.json"
    minsup_abs = math.ceil(min_support * n)  # Spark-compatible conversion

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[*]").config(
        "spark.sql.shuffle.partitions", "8").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    ref_sets = {}  # frozenset[int] -> freq
    for row in spark.read.parquet(os.path.join(args.ref_dir, "itemsets")).collect():
        ref_sets[frozenset(int(t) for t in row["items"])] = int(row["freq"])
    ref_rules = {}  # (frozenset[int], int) -> (supp, conf, lift)
    for row in spark.read.parquet(os.path.join(args.ref_dir, "rules")).collect():
        ant = frozenset(int(t) for t in row["antecedent"])
        cons = int(row["consequent"][0])
        ref_rules[(ant, cons)] = (float(row["support"]), float(row["confidence"]),
                                  float(row["lift"]))
    spark.stop()

    # fpm run: depth cap only prunes itemsets longer than A+1; the reference
    # comparison below restricts to rules with |antecedent| <= A on both sides.
    max_len = None if args.max_antecedents > 10**8 else args.max_antecedents + 1
    if max_len is None:
        max_len = len(np.unique(keys))
    result = fpm.mine_itemsets(offsets, keys, minsup_abs, max_len, args.threads)
    got_sets = {}
    for i in range(len(result["counts"])):
        iset = frozenset(result["itemset_keys"][result["itemset_offsets"][i]:
                                                 result["itemset_offsets"][i + 1]].tolist())
        got_sets[iset] = int(result["counts"][i])

    rp = fpm  # shorthand for the mine() call below
    rules = rp.mine(offsets, keys, min_support_abs=minsup_abs, max_len=max_len,
                    max_antecedents=args.max_antecedents, min_confidence=min_confidence,
                    threads=args.threads, k_witnesses=args.k_witnesses)
    got_rules = {}
    for i in range(len(rules["consequent"])):
        ant = frozenset(rules["ant_keys"][rules["ant_offsets"][i]:
                                        rules["ant_offsets"][i + 1]].tolist())
        got_rules[(ant, int(rules["consequent"][i]))] = (
            float(rules["support"][i]), float(rules["confidence"][i]),
            float(rules["lift"][i]), int(rules["n_baskets"][i]))

    # Spark has no antecedent cap; restrict its rules to |ant| <= A, and Spark
    # never emits rules for itemsets longer than its own search — both sides
    # must also respect the same mining depth.
    ref_rules_cap = {k: v for k, v in ref_rules.items()
                     if len(k[0]) <= args.max_antecedents and len(k[0]) + 1 <= max_len}

    missing = set(ref_rules_cap) - set(got_rules)
    extra = set(got_rules) - set(ref_rules_cap)
    metric_bad = []
    baskets_bad = []
    for k in set(ref_rules_cap) & set(got_rules):
        rs, gs = ref_rules_cap[k], got_rules[k]
        if not (rel_close(rs[0], gs[0]) and rel_close(rs[1], gs[1])
                and rel_close(rs[2], gs[2])):
            metric_bad.append((k, rs[:3], gs[:3]))
        # n_baskets exact: union itemset freq from the Spark itemset table
        union = k[0] | {k[1]}
        if union in ref_sets and ref_sets[union] != gs[3]:
            baskets_bad.append((k, ref_sets[union], gs[3]))

    # Itemset-level parity (counts exact), restricted to the mined depth.
    ref_sets_cap = {k: v for k, v in ref_sets.items() if len(k) <= max_len}
    missing_sets = set(ref_sets_cap) - set(got_sets)
    extra_sets = set(got_sets) - set(ref_sets_cap)
    bad_sets = [k for k in set(ref_sets_cap) & set(got_sets)
                if ref_sets_cap[k] != got_sets[k]]

    # Boundary audit around the support threshold.
    boundary_bad = 0
    for iset, cnt in got_sets.items():
        if cnt < minsup_abs:
            boundary_bad += 1
    for iset, cnt in ref_sets_cap.items():
        if cnt >= minsup_abs and iset not in got_sets:
            boundary_bad += 1

    print(f"itemsets: {len(got_sets)} fpm vs {len(ref_sets_cap)} spark — "
          f"{len(missing_sets)} missing, {len(extra_sets)} extra, "
          f"{len(bad_sets)} count mismatches")
    print(f"rules:    {len(got_rules)} fpm vs {len(ref_rules_cap)} spark — "
          f"{len(missing)} missing, {len(extra)} extra")
    print(f"metrics:  {len(metric_bad)} beyond 1e-9 relative")
    print(f"n_baskets: {len(baskets_bad)} mismatches vs exact Spark freq")
    print(f"boundary audit: {boundary_bad} threshold violations")

    # Witness validity against the raw dataset.
    witness_bad = 0
    if args.k_witnesses > 0:
        tx_items = []
        for i in range(n):
            tx_items.append(set(keys[offsets[i]:offsets[i + 1]].tolist()))
        k_wit = args.k_witnesses
        for i in range(len(rules["consequent"])):
            items = set(rules["ant_keys"][rules["ant_offsets"][i]:
                                        rules["ant_offsets"][i + 1]].tolist())
            items.add(int(rules["consequent"][i]))
            for w in rules["witnesses"][i]:
                if w == -1:
                    continue
                if not (0 <= w < n) or not items <= tx_items[w]:
                    witness_bad += 1
    print(f"witnesses: {witness_bad} invalid entries")

    ok = (not missing_sets and not extra_sets and not bad_sets and not missing
          and not extra and not metric_bad and not baskets_bad and not boundary_bad
          and not witness_bad)
    if not ok:
        for label, bad in [("missing rules", missing), ("extra rules", extra),
                           ("metric diffs", [b[0] for b in metric_bad])]:
            if bad:
                sample = sorted(bad, key=repr)[:5]
                print(f"  sample {label}: {sample}", file=sys.stderr)
        sys.exit("PARITY FAILURE")
    print("PARITY OK")


if __name__ == "__main__":
    main()
