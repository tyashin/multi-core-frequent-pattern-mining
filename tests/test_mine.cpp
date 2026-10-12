#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "bitset.h"
#include "mine.h"

namespace {

using ItemsetMap = std::map<std::vector<uint64_t>, uint64_t>;

// Independent oracle #1: inverted lists + std::set_intersection with
// anti-monotonic pruning. Shares no code with the bitset engine.
ItemsetMap oracle_mine(const std::vector<uint64_t>& offsets,
                       const std::vector<uint64_t>& keys, int64_t minsup, int max_len) {
  const size_t n = offsets.size() - 1;
  std::map<uint64_t, std::vector<uint32_t>> lists;
  for (size_t tx = 0; tx < n; ++tx) {
    std::vector<uint64_t> seen(keys.begin() + offsets[tx], keys.begin() + offsets[tx + 1]);
    std::sort(seen.begin(), seen.end());
    seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
    for (uint64_t it : seen) lists[it].push_back(static_cast<uint32_t>(tx));
  }
  std::vector<uint64_t> items;
  for (const auto& kv : lists) items.push_back(kv.first);

  ItemsetMap out;
  std::vector<uint64_t> cur;
  std::vector<uint32_t> all(n);
  for (size_t i = 0; i < n; ++i) all[i] = static_cast<uint32_t>(i);

  std::function<void(size_t, const std::vector<uint32_t>&, int)> rec =
      [&](size_t start, const std::vector<uint32_t>& base, int depth) {
        for (size_t i = start; i < items.size(); ++i) {
          std::vector<uint32_t> inter;
          const auto& l = lists[items[i]];
          std::set_intersection(base.begin(), base.end(), l.begin(), l.end(),
                                std::back_inserter(inter));
          if (static_cast<int64_t>(inter.size()) < minsup) continue;
          cur.push_back(items[i]);
          out[cur] = inter.size();
          if (depth < max_len) rec(i + 1, inter, depth + 1);
          cur.pop_back();
        }
      };
  rec(0, all, 1);
  return out;
}

// Independent oracle #2: full power-set enumeration with per-transaction
// membership tests, no pruning. Only feasible for few distinct items.
ItemsetMap oracle_powerset(const std::vector<uint64_t>& offsets,
                           const std::vector<uint64_t>& keys, int64_t minsup, int max_len) {
  const size_t n = offsets.size() - 1;
  std::set<uint64_t> uniq(keys.begin(), keys.end());
  std::vector<uint64_t> items(uniq.begin(), uniq.end());
  std::vector<std::set<uint64_t>> txs(n);
  for (size_t tx = 0; tx < n; ++tx)
    for (size_t k = offsets[tx]; k < offsets[tx + 1]; ++k) txs[tx].insert(keys[k]);

  ItemsetMap out;
  const size_t m = items.size();
  for (uint64_t mask = 1; mask < (uint64_t(1) << m); ++mask) {
    if (__builtin_popcountll(mask) > max_len) continue;
    uint64_t c = 0;
    for (size_t tx = 0; tx < n; ++tx) {
      bool all = true;
      for (size_t i = 0; i < m && all; ++i)
        if (mask & (uint64_t(1) << i)) all = txs[tx].count(items[i]) != 0;
      c += all;
    }
    if (static_cast<int64_t>(c) >= minsup) {
      std::vector<uint64_t> is;
      for (size_t i = 0; i < m; ++i)
        if (mask & (uint64_t(1) << i)) is.push_back(items[i]);
      out[is] = c;
    }
  }
  return out;
}

// Library output -> map with sorted itemset keys.
ItemsetMap as_map(const fpm::FrequentItemsets& fi) {
  ItemsetMap m;
  for (size_t i = 0; i + 1 < fi.offsets.size(); ++i) {
    std::vector<uint64_t> is(fi.ids.begin() + fi.offsets[i], fi.ids.begin() + fi.offsets[i + 1]);
    std::sort(is.begin(), is.end());
    EXPECT_TRUE(m.emplace(is, fi.counts[i]).second) << "duplicate itemset emitted";
    EXPECT_EQ(m[is], fi.counts[i]);
  }
  return m;
}

void expect_well_formed(const fpm::FrequentItemsets& fi, uint64_t n) {
  ASSERT_FALSE(fi.offsets.empty());
  EXPECT_EQ(fi.offsets[0], 0u);
  EXPECT_EQ(fi.offsets.size(), fi.counts.size() + 1);
  EXPECT_EQ(fi.offsets.back(), fi.ids.size());
  EXPECT_EQ(fi.n_transactions, n);
  for (size_t i = 1; i < fi.offsets.size(); ++i)
    EXPECT_GT(fi.offsets[i], fi.offsets[i - 1]);  // no empty itemsets
}

struct Csr {
  std::vector<uint64_t> offsets;
  std::vector<uint64_t> keys;
};

Csr make_random_csr(std::mt19937_64& rng, size_t n, const std::vector<uint64_t>& items,
                    const std::vector<double>& probs, bool with_duplicates = false) {
  Csr csr;
  csr.offsets.push_back(0);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (size_t tx = 0; tx < n; ++tx) {
    const size_t tx_start = csr.keys.size();
    for (size_t i = 0; i < items.size(); ++i)
      if (u(rng) < probs[i]) csr.keys.push_back(items[i]);
    if (with_duplicates && csr.keys.size() > tx_start) {
      // Duplicate the first item of this transaction a couple of times.
      const uint64_t dup = csr.keys[tx_start];
      csr.keys.push_back(dup);
      csr.keys.push_back(dup);
    }
    csr.offsets.push_back(csr.keys.size());
  }
  return csr;
}

fpm::FrequentItemsets run_mine(const Csr& csr, int64_t minsup, int max_len, int threads = 0,
                               bool debug_store = false) {
  auto idx = fpm::VerticalIndex::build(csr.offsets.data(), csr.offsets.size(), csr.keys.data(),
                                       csr.keys.size(), minsup, threads);
  return fpm::mine_itemsets(idx, max_len, threads, debug_store);
}

TEST(Mine, HandcraftedTiny) {
  Csr csr{{0, 3, 5, 7, 9, 13}, {1, 2, 3, 1, 2, 1, 3, 2, 3, 1, 2, 3, 4}};
  auto fi = run_mine(csr, 2, 4);
  expect_well_formed(fi, 5);
  ItemsetMap m = as_map(fi);
  EXPECT_EQ(m, oracle_mine(csr.offsets, csr.keys, 2, 4));
  // Spot checks.
  EXPECT_EQ(m.at({1}), 4u);
  EXPECT_EQ(m.at({1, 2}), 3u);
  EXPECT_EQ(m.at({1, 2, 3}), 2u);
  EXPECT_EQ(m.count({4}), 0u);      // support 1 < minsup
  EXPECT_EQ(m.count({1, 4}), 0u);   // pruned by anti-monotonicity
}

TEST(Mine, RandomVsPowersetOracle) {
  std::mt19937_64 rng(2024);
  // Non-contiguous, large, and zero item ids: ids are opaque.
  const std::vector<uint64_t> items = {0, 7, 13, 100, 1001, 1ull << 20,
                                       (1ull << 40) + 5, (1ull << 63), 42, 5, 6, 9999};
  const std::vector<double> probs = {0.5, 0.4, 0.35, 0.3, 0.25, 0.2,
                                     0.15, 0.1, 0.08, 0.05, 0.02, 0.01};
  Csr csr = make_random_csr(rng, 100, items, probs);
  for (int max_len : {1, 2, 3, 4}) {
    auto fi = run_mine(csr, 3, max_len, /*threads=*/1);
    expect_well_formed(fi, 100);
    EXPECT_EQ(as_map(fi), oracle_powerset(csr.offsets, csr.keys, 3, max_len))
        << "max_len=" << max_len;
  }
}

TEST(Mine, RandomVsInvertedListOracle) {
  std::mt19937_64 rng(77);
  std::vector<uint64_t> items(100);
  std::vector<double> probs(100);
  for (int i = 0; i < 100; ++i) {
    items[i] = static_cast<uint64_t>(i) * 3;
    probs[i] = i < 20 ? 0.05 : (i < 60 ? 0.01 : 0.002);
  }
  Csr csr = make_random_csr(rng, 5000, items, probs);
  auto fi = run_mine(csr, 10, 5);
  expect_well_formed(fi, 5000);
  EXPECT_EQ(as_map(fi), oracle_mine(csr.offsets, csr.keys, 10, 5));
}

TEST(Mine, SupportBoundaryPlusMinusOne) {
  std::mt19937_64 rng(8);
  const std::vector<uint64_t> items = {1, 2, 3, 4, 5, 6};
  const std::vector<double> probs = {0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
  Csr csr = make_random_csr(rng, 200, items, probs);
  auto oracle = oracle_powerset(csr.offsets, csr.keys, 1, 3);
  // For every oracle count, minsup = count keeps the itemset, count + 1 drops it.
  for (const auto& kv : oracle) {
    const int64_t c = static_cast<int64_t>(kv.second);
    auto kept = as_map(run_mine(csr, c, 3));
    auto dropped = as_map(run_mine(csr, c + 1, 3));
    EXPECT_EQ(kept.count(kv.first), 1u);
    EXPECT_EQ(dropped.count(kv.first), 0u);
    // And the two result sets differ exactly by itemsets with count == c.
    for (const auto& kv2 : oracle) {
      const bool expected = static_cast<int64_t>(kv2.second) >= c + 1;
      EXPECT_EQ(dropped.count(kv2.first) == 1, expected);
    }
  }
}

TEST(Mine, MaxLenExactness) {
  std::mt19937_64 rng(31);
  const std::vector<uint64_t> items = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<double> probs = {0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15};
  Csr csr = make_random_csr(rng, 500, items, probs);
  for (int max_len : {1, 2, 3, 5, 8, 100}) {
    auto fi = run_mine(csr, 4, max_len);
    const int cap = std::min<int>(max_len, 8);  // only 8 distinct items exist
    for (size_t i = 0; i + 1 < fi.offsets.size(); ++i)
      EXPECT_LE(fi.offsets[i + 1] - fi.offsets[i], static_cast<uint64_t>(cap));
    EXPECT_EQ(as_map(fi), oracle_mine(csr.offsets, csr.keys, 4, max_len));
  }
}

TEST(Mine, DuplicateItemsInTransaction) {
  std::mt19937_64 rng(41);
  const std::vector<uint64_t> items = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::vector<double> probs = {0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02};
  Csr csr = make_random_csr(rng, 300, items, probs, /*with_duplicates=*/true);
  auto fi = run_mine(csr, 3, 3);
  expect_well_formed(fi, 300);
  EXPECT_EQ(as_map(fi), oracle_mine(csr.offsets, csr.keys, 3, 3));
}

TEST(Mine, EdgeCases) {
  // No transactions at all.
  {
    Csr csr{{0}, {}};
    auto fi = run_mine(csr, 1, 3);
    expect_well_formed(fi, 0);
    EXPECT_TRUE(fi.counts.empty());
  }
  // All transactions empty.
  {
    Csr csr{{0, 0, 0, 0}, {}};
    auto fi = run_mine(csr, 1, 3);
    EXPECT_TRUE(fi.counts.empty());
  }
  // Single-item transactions only.
  {
    Csr csr{{0, 1, 2, 3}, {10, 10, 20}};
    auto fi = run_mine(csr, 1, 5);
    ItemsetMap m = as_map(fi);
    EXPECT_EQ(m.size(), 2u);
    EXPECT_EQ(m.at({10}), 2u);
    EXPECT_EQ(m.at({20}), 1u);
  }
  // min_support above N: nothing survives.
  {
    Csr csr{{0, 2, 4}, {1, 2, 1, 3}};
    auto fi = run_mine(csr, 3, 3);
    EXPECT_TRUE(fi.counts.empty());
  }
  // Empty suffix handling: single distinct item.
  {
    Csr csr{{0, 1, 2}, {7, 7}};
    auto fi = run_mine(csr, 1, 10);
    ItemsetMap m = as_map(fi);
    EXPECT_EQ(m.size(), 1u);
    EXPECT_EQ(m.at({7}), 2u);
  }
}

TEST(Mine, InvalidInputThrows) {
  Csr good{{0, 2}, {1, 2}};
  // offsets[0] != 0
  EXPECT_THROW(fpm::VerticalIndex::build(std::vector<uint64_t>{1, 2}.data(), 2,
                                         good.keys.data(), 2, 1, 1),
               std::invalid_argument);
  // non-monotone offsets
  EXPECT_THROW(fpm::VerticalIndex::build(std::vector<uint64_t>{0, 2, 1}.data(), 3,
                                         good.keys.data(), 2, 1, 1),
               std::invalid_argument);
  // offsets[N] != keys_len
  EXPECT_THROW(fpm::VerticalIndex::build(good.offsets.data(), 2, good.keys.data(), 5, 1, 1),
               std::invalid_argument);
  // min_support_abs < 1
  EXPECT_THROW(fpm::VerticalIndex::build(good.offsets.data(), 2, good.keys.data(), 2, 0, 1),
               std::invalid_argument);
  // max_len < 1
  auto idx = fpm::VerticalIndex::build(good.offsets.data(), 2, good.keys.data(), 2, 1, 1);
  EXPECT_THROW(fpm::mine_itemsets(idx, 0, 1), std::invalid_argument);
  // null offsets
  EXPECT_THROW(fpm::VerticalIndex::build(nullptr, 2, good.keys.data(), 2, 1, 1),
               std::invalid_argument);
}

TEST(Mine, DeterministicAcrossThreadCounts) {
  std::mt19937_64 rng(64);
  std::vector<uint64_t> items(60);
  std::vector<double> probs(60);
  for (int i = 0; i < 60; ++i) {
    items[i] = static_cast<uint64_t>(i + 1);
    probs[i] = i < 15 ? 0.08 : 0.02;
  }
  Csr csr = make_random_csr(rng, 8000, items, probs);
  auto a = run_mine(csr, 6, 4, /*threads=*/1);
  auto b = run_mine(csr, 6, 4, /*threads=*/4);
  EXPECT_EQ(a.ids, b.ids);
  EXPECT_EQ(a.offsets, b.offsets);
  EXPECT_EQ(a.counts, b.counts);
}

TEST(Mine, DeterministicAcrossIsas) {
  std::mt19937_64 rng(65);
  std::vector<uint64_t> items(40);
  std::vector<double> probs(40, 0.05);
  for (int i = 0; i < 40; ++i) items[i] = static_cast<uint64_t>(i + 1);
  Csr csr = make_random_csr(rng, 4000, items, probs);

  std::vector<fpm::Isa> isas{fpm::Isa::kScalar};
  if (fpm::detect_isa() >= fpm::Isa::kAvx2) isas.push_back(fpm::Isa::kAvx2);
  if (fpm::detect_isa() >= fpm::Isa::kAvx512) isas.push_back(fpm::Isa::kAvx512);

  fpm::force_isa(isas[0]);
  auto ref = run_mine(csr, 4, 4, /*threads=*/1);
  for (size_t i = 1; i < isas.size(); ++i) {
    fpm::force_isa(isas[i]);
    auto got = run_mine(csr, 4, 4, /*threads=*/1);
    EXPECT_EQ(ref.ids, got.ids) << fpm::isa_name(isas[i]);
    EXPECT_EQ(ref.counts, got.counts) << fpm::isa_name(isas[i]);
  }
  fpm::clear_forced_isa();
}

TEST(Mine, TerminalStoreSkipMatchesStore) {
  std::mt19937_64 rng(66);
  std::vector<uint64_t> items(30);
  std::vector<double> probs(30, 0.06);
  for (int i = 0; i < 30; ++i) items[i] = static_cast<uint64_t>(i + 1);
  Csr csr = make_random_csr(rng, 3000, items, probs);
  auto skip = run_mine(csr, 4, 3, /*threads=*/1, /*debug_store=*/false);
  auto store = run_mine(csr, 4, 3, /*threads=*/1, /*debug_store=*/true);
  EXPECT_EQ(skip.ids, store.ids);
  EXPECT_EQ(skip.offsets, store.offsets);
  EXPECT_EQ(skip.counts, store.counts);
}

TEST(Mine, DenseOfLookup) {
  Csr csr{{0, 2, 4}, {5, 9, 5, 3}};
  auto idx = fpm::VerticalIndex::build(csr.offsets.data(), csr.offsets.size(), csr.keys.data(),
                                       csr.keys.size(), 1, 1);
  EXPECT_GE(idx.dense_of(5), 0);
  EXPECT_GE(idx.dense_of(9), 0);
  EXPECT_GE(idx.dense_of(3), 0);
  EXPECT_EQ(idx.dense_of(12345), -1);
  // ids round-trip through dense index
  EXPECT_EQ(idx.item_id(static_cast<size_t>(idx.dense_of(9))), 9u);
}

}  // namespace
