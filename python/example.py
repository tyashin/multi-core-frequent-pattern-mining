"""Minimal fpm example: mine association rules from a CSR transaction database."""
import numpy as np

import fpm

# Transactions as item-id lists; ids are opaque uint64 values owned by the caller.
transactions = [
    [10, 20, 30],
    [10, 20],
    [10, 30],
    [20, 30],
    [10, 20, 30, 40],
]

offsets = np.zeros(len(transactions) + 1, dtype=np.uint64)
keys = np.array([i for tx in transactions for i in tx], dtype=np.uint64)
offsets[1:] = np.cumsum([len(tx) for tx in transactions])

result = fpm.mine(
    offsets,
    keys,
    min_support_abs=2,
    max_len=3,            # itemset depth cap: antecedent cap + 1
    max_antecedents=2,    # A
    min_confidence=0.5,   # global generation-time floor
    threads=0,            # 0 = auto-detect
    k_witnesses=3,        # witness transactions per rule
)

for i in range(len(result["consequent"])):
    ant = result["ant_keys"][result["ant_offsets"][i]:result["ant_offsets"][i + 1]]
    print(f"{list(ant)} -> {result['consequent'][i]}  "
          f"supp={result['support'][i]:.3f} conf={result['confidence'][i]:.3f} "
          f"lift={result['lift'][i]:.3f} n={result['n_baskets'][i]} "
          f"witnesses={result['witnesses'][i].tolist()}")
