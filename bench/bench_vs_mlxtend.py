"""Benchmark fpm vs mlxtend on FIMI datasets, with strict output parity.

Datasets: retail, kosarak, accidents from the FIMI repository mirror at
https://cs.rpi.edu/~zaki/Workshops/FIMI/data/ — fetch with:
  curl -sSLO https://cs.rpi.edu/~zaki/Workshops/FIMI/data/retail.dat.gz
  curl -sSLO https://cs.rpi.edu/~zaki/Workshops/FIMI/data/kosarak.dat.gz
  curl -sSLO https://cs.rpi.edu/~zaki/Workshops/FIMI/data/accidents.dat.gz
placed under bench/data/ (not committed to git). mlxtend is a dev-only oracle.

Usage:
  python bench/bench_vs_mlxtend.py --dataset retail --minsup 0.01
  python bench/bench_vs_mlxtend.py --all          # default suite
"""

import argparse
import gzip
import time

import numpy as np

import fpm


def load_fimi(path):
    offsets = [0]
    keys = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            keys.extend(int(tok) for tok in line.split())
            offsets.append(len(keys))
    return np.array(offsets, dtype=np.uint64), np.array(keys, dtype=np.uint64)


def fpm_itemsets(offsets, keys, minsup_abs, max_len, threads):
    t0 = time.perf_counter()
    r = fpm.mine_itemsets(offsets, keys, minsup_abs, max_len, threads)
    dt = time.perf_counter() - t0
    out = {}
    for i in range(len(r["counts"])):
        iset = frozenset(r["itemset_keys"][r["itemset_offsets"][i]:
                                           r["itemset_offsets"][i + 1]].tolist())
        out[iset] = int(r["counts"][i])
    return out, dt


def mlxtend_itemsets(offsets, keys, minsup_frac, max_len):
    import pandas as pd
    from mlxtend.frequent_patterns import fpgrowth
    from mlxtend.preprocessing import TransactionEncoder

    n = len(offsets) - 1
    t0 = time.perf_counter()
    te = TransactionEncoder()
    txs = [keys[offsets[i]:offsets[i + 1]].tolist() for i in range(n)]
    arr = te.fit_transform(txs, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(arr, columns=te.columns_)
    # mlxtend/pandas sparse limitation: integer column names must start at 0.
    df.columns = [str(c) for c in df.columns]
    prep_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = fpgrowth(df, min_support=minsup_frac, use_colnames=True, max_len=max_len)
    dt = time.perf_counter() - t0
    out = {frozenset(int(c) for c in s): int(round(supp * n)) for s, supp in
           zip(res["itemsets"], res["support"])}
    return out, dt, prep_s


def check_parity(a, b, label):
    if a != b:
        missing = set(b) - set(a)
        extra = set(a) - set(b)
        bad_count = [k for k in set(a) & set(b) if a[k] != b[k]]
        raise AssertionError(
            f"{label}: parity failure — {len(missing)} missing, {len(extra)} extra, "
            f"{len(bad_count)} count mismatches (e.g. {bad_count[:3]})")
    print(f"  parity OK ({len(a)} itemsets, 0 missing / 0 extra / counts exact)")


def run(dataset, minsup_frac, max_len, threads, skip_mlxtend):
    path = f"bench/data/{dataset}.dat.gz"
    offsets, keys = load_fimi(path)
    n = len(offsets) - 1
    minsup_abs = int(np.ceil(minsup_frac * n))  # Spark-compatible conversion
    distinct = len(np.unique(keys))
    # fpm needs a concrete cap; "no cap" = number of frequent items.
    max_len_eff = max_len if max_len is not None else distinct
    print(f"{dataset}: N={n}, M={len(keys)}, distinct={distinct}, "
          f"minsup {minsup_frac} -> {minsup_abs}, max_len={max_len_eff}")

    fr, fdt = fpm_itemsets(offsets, keys, minsup_abs, max_len_eff, threads)
    print(f"  fpm:     {len(fr):>9} itemsets in {fdt * 1000:9.1f} ms "
          f"(threads={threads}, isa={fpm.isa_in_use()})")

    if not skip_mlxtend:
        mr, mdt, prep = mlxtend_itemsets(offsets, keys, minsup_frac, max_len)
        print(f"  mlxtend: {len(mr):>9} itemsets in {mdt * 1000:9.1f} ms "
              f"(one-hot prep {prep * 1000:.0f} ms excluded)")
        check_parity(fr, mr, dataset)
        print(f"  speedup: {mdt / fdt:6.1f}x")
    return fdt


SUITE = [
    # dataset, min_support fraction, max_len, run mlxtend?
    ("accidents", 0.2, None, True),
    ("accidents", 0.1, None, True),
    ("retail", 0.01, None, True),
    ("retail", 0.005, None, True),
    ("kosarak", 0.05, None, False),   # mlxtend one-hot is intractable here
    ("kosarak", 0.02, None, False),
    ("kosarak", 0.01, 8, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run the default suite")
    ap.add_argument("--dataset", choices=["retail", "kosarak", "accidents"])
    ap.add_argument("--minsup", type=float)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--skip-mlxtend", action="store_true")
    args = ap.parse_args()

    runs = SUITE if args.all or not args.dataset else [
        (args.dataset, args.minsup or 0.01, args.max_len, not args.skip_mlxtend)]
    for dataset, frac, max_len, use_mlxtend in runs:
        run(dataset, frac, max_len, args.threads, skip_mlxtend=not use_mlxtend)
        print()


if __name__ == "__main__":
    main()
