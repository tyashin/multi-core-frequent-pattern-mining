"""Integration tests for the fpm Python module (pure-Python reference oracle)."""
import itertools

import numpy as np
import pytest

import fpm


def csr_from_transactions(txs):
    offsets = [0]
    keys = []
    for tx in txs:
        keys.extend(tx)
        offsets.append(len(keys))
    return np.array(offsets, dtype=np.uint64), np.array(keys, dtype=np.uint64)


def reference_rules(txs, minsup, max_len, max_antecedents, min_confidence,
                    supp_thr=(), conf_thr=(), lift_thr=()):
    """Naive reference: full subset enumeration, per-tx membership counting."""
    n = len(txs)
    tx_sets = [set(tx) for tx in txs]
    items = sorted({i for tx in tx_sets for i in tx})
    counts = {}

    def count(itemset):
        if itemset not in counts:
            counts[itemset] = sum(1 for t in tx_sets if set(itemset) <= t)
        return counts[itemset]

    rules = {}
    for size in range(2, max_len + 1):
        for z in itertools.combinations(items, size):
            cz = count(z)
            if cz < minsup or len(z) - 1 > max_antecedents:
                continue
            size_key = min(len(z), 5) - 1
            supp = cz / n
            if supp_thr and size_key < len(supp_thr) and supp < supp_thr[size_key]:
                continue
            for c in z:
                x = tuple(i for i in z if i != c)
                conf = cz / count(x)
                floor = min_confidence
                if conf_thr and size_key < len(conf_thr):
                    floor = max(floor, conf_thr[size_key])
                if conf < floor:
                    continue
                lift = conf / (count((c,)) / n)
                if lift_thr and size_key < len(lift_thr) and lift < lift_thr[size_key]:
                    continue
                rules[(frozenset(x), c)] = (supp, conf, lift, cz)
    return rules


def rules_as_dict(result):
    out = {}
    for i in range(len(result["consequent"])):
        ant = result["ant_keys"][result["ant_offsets"][i]:result["ant_offsets"][i + 1]]
        out[(frozenset(ant.tolist()), int(result["consequent"][i]))] = (
            result["support"][i], result["confidence"][i], result["lift"][i],
            int(result["n_baskets"][i]))
    return out


def random_transactions(rng, n, num_items, p_hi, p_lo, hi_count):
    return [[i for i in range(num_items)
             if rng.random() < (p_hi if i < hi_count else p_lo)]
            for _ in range(n)]


def test_matches_reference():
    rng = np.random.default_rng(11)
    txs = random_transactions(rng, 400, 10, 0.25, 0.05, 4)
    offsets, keys = csr_from_transactions(txs)
    result = fpm.mine(offsets, keys, min_support_abs=5, max_len=3, max_antecedents=2,
                      min_confidence=0.3)
    got = rules_as_dict(result)
    exp = reference_rules(txs, 5, 3, 2, 0.3)
    assert got.keys() == exp.keys()
    for k in exp:
        assert got[k][0] == pytest.approx(exp[k][0], rel=1e-12)
        assert got[k][1] == pytest.approx(exp[k][1], rel=1e-12)
        assert got[k][2] == pytest.approx(exp[k][2], rel=1e-12)
        assert got[k][3] == exp[k][3]


def test_dtypes_and_shapes():
    offsets, keys = csr_from_transactions([[1, 2], [1, 2], [1, 3]])
    r = fpm.mine(offsets, keys, min_support_abs=1, max_len=2, max_antecedents=1)
    assert r["ant_offsets"].dtype == np.uint64
    assert r["ant_keys"].dtype == np.uint64
    assert r["consequent"].dtype == np.uint64
    assert r["support"].dtype == np.float64
    assert r["confidence"].dtype == np.float64
    assert r["lift"].dtype == np.float64
    assert r["n_baskets"].dtype == np.int64
    n_rules = len(r["consequent"])
    assert r["ant_offsets"].shape == (n_rules + 1,)
    assert r["witnesses"].shape == (n_rules, 0)
    assert n_rules > 0


def test_witness_validity():
    rng = np.random.default_rng(5)
    txs = random_transactions(rng, 300, 8, 0.3, 0.1, 3)
    offsets, keys = csr_from_transactions(txs)
    k = 4
    r = fpm.mine(offsets, keys, min_support_abs=4, max_len=3, max_antecedents=2,
                 k_witnesses=k)
    tx_sets = [set(tx) for tx in txs]
    for i in range(len(r["consequent"])):
        items = set(r["ant_keys"][r["ant_offsets"][i]:r["ant_offsets"][i + 1]].tolist())
        items.add(int(r["consequent"][i]))
        wit = [w for w in r["witnesses"][i] if w != -1]
        assert len(wit) == min(k, r["n_baskets"][i])
        assert wit == sorted(wit)
        for w in wit:
            assert items <= tx_sets[w]


def test_per_length_thresholds():
    rng = np.random.default_rng(21)
    txs = random_transactions(rng, 500, 9, 0.2, 0.05, 4)
    offsets, keys = csr_from_transactions(txs)
    supp_thr = np.array([0.0, 0.0, 0.02])
    conf_thr = np.array([0.0, 0.0, 0.8])
    r = fpm.mine(offsets, keys, min_support_abs=4, max_len=3, max_antecedents=2,
                 supp_thr=supp_thr, conf_thr=conf_thr)
    got = rules_as_dict(r)
    exp = reference_rules(txs, 4, 3, 2, 0.0, supp_thr=tuple(supp_thr),
                          conf_thr=tuple(conf_thr))
    assert got.keys() == exp.keys()


def test_threads_determinism():
    rng = np.random.default_rng(31)
    txs = random_transactions(rng, 2000, 20, 0.1, 0.03, 6)
    offsets, keys = csr_from_transactions(txs)
    a = fpm.mine(offsets, keys, min_support_abs=6, max_len=3, max_antecedents=2,
                 min_confidence=0.2, threads=1, k_witnesses=3)
    b = fpm.mine(offsets, keys, min_support_abs=6, max_len=3, max_antecedents=2,
                 min_confidence=0.2, threads=4, k_witnesses=3)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_empty_database():
    offsets, keys = csr_from_transactions([])
    r = fpm.mine(offsets, keys, min_support_abs=1, max_len=3, max_antecedents=2)
    assert len(r["consequent"]) == 0
    assert r["ant_offsets"].tolist() == [0]


def test_invalid_inputs():
    offsets, keys = csr_from_transactions([[1, 2], [1, 2]])
    with pytest.raises(ValueError):
        fpm.mine(offsets, keys, min_support_abs=0, max_len=2, max_antecedents=1)
    with pytest.raises(ValueError):
        fpm.mine(offsets, keys, min_support_abs=1, max_len=0, max_antecedents=1)
    with pytest.raises(ValueError):
        fpm.mine(offsets, keys, min_support_abs=1, max_len=2, max_antecedents=0)
    with pytest.raises(ValueError):  # offsets[N] != len(keys)
        fpm.mine(offsets, np.array([1], dtype=np.uint64), min_support_abs=1, max_len=2,
                 max_antecedents=1)
    with pytest.raises(ValueError):  # threshold array too long
        fpm.mine(offsets, keys, min_support_abs=1, max_len=2, max_antecedents=1,
                 conf_thr=np.zeros(6))
    with pytest.raises(TypeError):  # wrong dtype (float keys)
        fpm.mine(offsets, keys.astype(np.float64), min_support_abs=1, max_len=2,
                 max_antecedents=1)


def test_isa_helpers():
    assert fpm.detect_isa() in ("scalar", "avx2", "avx512")
    assert fpm.isa_in_use() in ("scalar", "avx2", "avx512")
