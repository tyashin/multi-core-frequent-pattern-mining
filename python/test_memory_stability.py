"""RSS-stability test: repeated fpm.mine calls must not grow memory.

LeakSanitizer (in the ASan CI config) catches leaks that are unreachable at
exit; this test catches the complementary class — memory that stays reachable
but grows per call (e.g. a growing static cache) — the leaks that matter when
the library is embedded in a long-running host.
"""

import numpy as np

import fpm


def _rss_kb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS not found")


def _make_csr(rng, n=30000, num_items=150):
    offsets = [0]
    keys = []
    for _ in range(n):
        keys.extend(i for i in range(num_items)
                    if rng.random() < (0.06 if i < 40 else 0.01))
        offsets.append(len(keys))
    return np.array(offsets, dtype=np.uint64), np.array(keys, dtype=np.uint64)


def test_rss_plateaus_over_repeated_mine_calls():
    rng = np.random.default_rng(3)
    offsets, keys = _make_csr(rng)
    rss = []
    iterations = 120
    for it in range(iterations):
        r = fpm.mine(offsets, keys, min_support_abs=20, max_len=3, max_antecedents=2,
                     min_confidence=0.0, threads=2, k_witnesses=3)
        assert len(r["consequent"]) > 0
        del r
        if it % 10 == 9:
            rss.append(_rss_kb())

    warmup, late = rss[1], rss[-1]  # skip allocator warm-up
    growth_kb = late - warmup
    limit_kb = max(16384, int(0.05 * warmup))  # 16 MB or 5%, whichever is larger
    assert growth_kb < limit_kb, (
        f"RSS grew {growth_kb} kB ({warmup} -> {late} kB) over {iterations} iterations; "
        f"per-call samples: {rss}")
