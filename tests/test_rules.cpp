#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "mine.h"
#include "rules.h"
#include "witness.h"

namespace {

using ItemsetMap = std::map<std::vector<uint64_t>, uint64_t>;

// Same inverted-list oracle as the mine tests: gives the reference itemset
// counts that the rule oracle filters independently.
ItemsetMap oracle_itemsets(const std::vector<uint64_t>& offsets,
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

struct OracleRule {
  std::vector<uint64_t> ant;  // sorted
  uint64_t cons;
  double supp, conf, lift;
  int64_t n_baskets;
  bool operator<(const OracleRule& o) const {
    if (ant != o.ant) return ant < o.ant;
    return cons < o.cons;
  }
};

std::vector<OracleRule> oracle_rules(const ItemsetMap& sets, uint64_t n, int max_antecedents,
                                     double min_confidence, const fpm::Thresholds& thr) {
  std::vector<OracleRule> out;
  for (const auto& kv : sets) {
    const std::vector<uint64_t>& z = kv.first;
    if (z.size() < 2) continue;
    if (static_cast<int>(z.size()) - 1 > max_antecedents) continue;
    const uint64_t cnt_z = kv.second;
    const int size_key = std::min<int>(static_cast<int>(z.size()), fpm::Thresholds::kSizeKeyMax);
    const double supp = static_cast<double>(cnt_z) / static_cast<double>(n);
    if (supp < thr.supp_at(size_key)) continue;
    for (uint64_t c : z) {
      std::vector<uint64_t> x;
      for (uint64_t it : z)
        if (it != c) x.push_back(it);
      const uint64_t cnt_x = sets.at(x);
      const uint64_t cnt_c = sets.at({c});
      const double conf = static_cast<double>(cnt_z) / static_cast<double>(cnt_x);
      if (conf < std::max(min_confidence, thr.conf_at(size_key))) continue;
      const double lift = conf * static_cast<double>(n) / static_cast<double>(cnt_c);
      if (lift < thr.lift_at(size_key)) continue;
      out.push_back({x, c, supp, conf, lift, static_cast<int64_t>(cnt_z)});
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<OracleRule> as_oracle_rules(const fpm::Rules& r) {
  std::vector<OracleRule> out;
  for (size_t i = 0; i < r.size(); ++i) {
    std::vector<uint64_t> ant(r.ant_keys.begin() + r.ant_offsets[i],
                              r.ant_keys.begin() + r.ant_offsets[i + 1]);
    std::sort(ant.begin(), ant.end());
    out.push_back({ant, r.consequent[i], r.support[i], r.confidence[i], r.lift[i],
                   r.n_baskets[i]});
  }
  std::sort(out.begin(), out.end());
  return out;
}

void expect_rules_eq(const std::vector<OracleRule>& got, const std::vector<OracleRule>& exp) {
  ASSERT_EQ(got.size(), exp.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i].ant, exp[i].ant) << "rule " << i;
    EXPECT_EQ(got[i].cons, exp[i].cons) << "rule " << i;
    EXPECT_NEAR(got[i].supp, exp[i].supp, 1e-12);
    EXPECT_NEAR(got[i].conf, exp[i].conf, 1e-12);
    EXPECT_NEAR(got[i].lift, exp[i].lift, 1e-12);
    EXPECT_EQ(got[i].n_baskets, exp[i].n_baskets);
  }
}

struct Csr {
  std::vector<uint64_t> offsets;
  std::vector<uint64_t> keys;
};

Csr make_random_csr(std::mt19937_64& rng, size_t n, size_t num_items, double p_hi, double p_lo,
                    size_t hi_count) {
  Csr csr;
  csr.offsets.push_back(0);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (size_t tx = 0; tx < n; ++tx) {
    for (size_t i = 0; i < num_items; ++i)
      if (u(rng) < (i < hi_count ? p_hi : p_lo)) csr.keys.push_back(i + 1);
    csr.offsets.push_back(csr.keys.size());
  }
  return csr;
}

fpm::Rules run_rules(const Csr& csr, int64_t minsup, int max_len, const fpm::RuleParams& p) {
  auto idx = fpm::VerticalIndex::build(csr.offsets.data(), csr.offsets.size(), csr.keys.data(),
                                       csr.keys.size(), minsup, p.threads);
  auto fi = fpm::mine_itemsets(idx, max_len, p.threads);
  return fpm::generate_rules(idx, fi, p);
}

TEST(Rules, HandcraftedMetrics) {
  // N = 5; {1,2}: 3, {1,2,3}: 2, {1}: 4, {3}: 4.
  Csr csr{{0, 3, 5, 7, 9, 13}, {1, 2, 3, 1, 2, 1, 3, 2, 3, 1, 2, 3, 4}};
  fpm::RuleParams p;
  p.max_antecedents = 3;
  p.min_confidence = 0.0;
  auto rules = run_rules(csr, 2, 4, p);
  auto got = as_oracle_rules(rules);
  // Rule {1,2} -> {3}: conf = 2/3, supp = 2/5, lift = (2/3)/(4/5) = 5/6.
  bool found = false;
  for (const auto& r : got) {
    if (r.ant == std::vector<uint64_t>{1, 2} && r.cons == 3) {
      found = true;
      EXPECT_DOUBLE_EQ(r.conf, 2.0 / 3.0);
      EXPECT_DOUBLE_EQ(r.supp, 2.0 / 5.0);
      EXPECT_DOUBLE_EQ(r.lift, 5.0 / 6.0);
      EXPECT_EQ(r.n_baskets, 2);
    }
  }
  EXPECT_TRUE(found);
  // Confidence-1.0 rules: {1,2,3} -> {4} impossible (4 not frequent); check
  // {3} -> {1}: conf = 3/4.
  bool found2 = false;
  for (const auto& r : got)
    if (r.ant == std::vector<uint64_t>{3} && r.cons == 1) {
      found2 = true;
      EXPECT_DOUBLE_EQ(r.conf, 3.0 / 4.0);
    }
  EXPECT_TRUE(found2);
}

TEST(Rules, MatchesOracleSweep) {
  std::mt19937_64 rng(123);
  Csr csr = make_random_csr(rng, 2000, 25, 0.08, 0.02, 8);
  const int64_t minsup = 8;
  const int max_len = 4;
  auto sets = oracle_itemsets(csr.offsets, csr.keys, minsup, max_len);

  // Baseline: no thresholds.
  {
    fpm::RuleParams p;
    p.max_antecedents = 3;
    p.min_confidence = 0.0;
    expect_rules_eq(as_oracle_rules(run_rules(csr, minsup, max_len, p)),
                    oracle_rules(sets, 2000, 3, 0.0, p.thr));
  }
  // Global min_confidence.
  {
    fpm::RuleParams p;
    p.max_antecedents = 3;
    p.min_confidence = 0.6;
    expect_rules_eq(as_oracle_rules(run_rules(csr, minsup, max_len, p)),
                    oracle_rules(sets, 2000, 3, 0.6, p.thr));
  }
  // Per-length thresholds on every metric and every size key.
  {
    fpm::RuleParams p;
    p.max_antecedents = 3;
    p.min_confidence = 0.3;
    p.thr.supp = {0.0, 0.005, 0.01, 0.02};
    p.thr.conf = {0.0, 0.4, 0.7, 0.9};
    p.thr.lift = {0.0, 1.0, 1.5, 2.0};
    expect_rules_eq(as_oracle_rules(run_rules(csr, minsup, max_len, p)),
                    oracle_rules(sets, 2000, 3, 0.3, p.thr));
  }
  // Antecedent cap.
  {
    fpm::RuleParams p;
    p.max_antecedents = 1;
    expect_rules_eq(as_oracle_rules(run_rules(csr, minsup, max_len, p)),
                    oracle_rules(sets, 2000, 1, 0.0, p.thr));
  }
}

TEST(Rules, SparkCompatMaxCombination) {
  // conf_thr below min_confidence: min_confidence wins; above: conf_thr wins.
  std::mt19937_64 rng(9);
  Csr csr = make_random_csr(rng, 1500, 20, 0.09, 0.02, 6);
  const int64_t minsup = 6;
  const int max_len = 3;
  auto sets = oracle_itemsets(csr.offsets, csr.keys, minsup, max_len);

  fpm::RuleParams p;
  p.max_antecedents = 2;
  p.min_confidence = 0.5;
  p.thr.conf = {0.0, 0.2, 0.8};  // size 2: 0.2 (< 0.5), size >= 3: 0.8 (> 0.5)
  expect_rules_eq(as_oracle_rules(run_rules(csr, minsup, max_len, p)),
                  oracle_rules(sets, 1500, 2, 0.5, p.thr));
}

TEST(Rules, DeterministicAcrossThreads) {
  std::mt19937_64 rng(55);
  Csr csr = make_random_csr(rng, 6000, 40, 0.06, 0.015, 10);
  fpm::RuleParams p;
  p.max_antecedents = 3;
  p.min_confidence = 0.4;
  p.k_witnesses = 5;

  p.threads = 1;
  auto a = run_rules(csr, 10, 4, p);
  p.threads = 4;
  auto b = run_rules(csr, 10, 4, p);
  EXPECT_EQ(a.ant_offsets, b.ant_offsets);
  EXPECT_EQ(a.ant_keys, b.ant_keys);
  EXPECT_EQ(a.consequent, b.consequent);
  EXPECT_EQ(a.support, b.support);
  EXPECT_EQ(a.confidence, b.confidence);
  EXPECT_EQ(a.lift, b.lift);
  EXPECT_EQ(a.n_baskets, b.n_baskets);
  EXPECT_EQ(a.witnesses, b.witnesses);
}

TEST(Rules, WitnessValidity) {
  std::mt19937_64 rng(77);
  Csr csr = make_random_csr(rng, 1000, 15, 0.15, 0.05, 5);
  fpm::RuleParams p;
  p.max_antecedents = 2;
  p.min_confidence = 0.3;
  p.k_witnesses = 5;
  p.threads = 2;
  auto rules = run_rules(csr, 5, 3, p);
  ASSERT_FALSE(rules.consequent.empty());
  ASSERT_EQ(rules.witnesses.size(), rules.size() * 5);

  // Per-transaction item sets for validity checking.
  const size_t n = csr.offsets.size() - 1;
  std::vector<std::set<uint64_t>> txs(n);
  for (size_t tx = 0; tx < n; ++tx)
    for (size_t k = csr.offsets[tx]; k < csr.offsets[tx + 1]; ++k) txs[tx].insert(csr.keys[k]);

  for (size_t r = 0; r < rules.size(); ++r) {
    std::set<uint64_t> items(rules.ant_keys.begin() + rules.ant_offsets[r],
                             rules.ant_keys.begin() + rules.ant_offsets[r + 1]);
    items.insert(rules.consequent[r]);
    int found = 0;
    for (int j = 0; j < 5; ++j) {
      const int32_t w = rules.witnesses[r * 5 + j];
      if (w == -1) continue;
      ++found;
      ASSERT_GE(w, 0);
      ASSERT_LT(static_cast<size_t>(w), n);
      // The witness transaction verifiably contains every rule item.
      for (uint64_t it : items) EXPECT_TRUE(txs[w].count(it)) << "rule " << r;
    }
    // Exactly min(k, n_baskets) witnesses must be found.
    EXPECT_EQ(found, std::min<int64_t>(5, rules.n_baskets[r])) << "rule " << r;
    // Ascending transaction order within a rule's witnesses.
    for (int j = 1; j < 5; ++j) {
      if (rules.witnesses[r * 5 + j] != -1) {
        EXPECT_GT(rules.witnesses[r * 5 + j], rules.witnesses[r * 5 + j - 1]);
      }
    }
  }
}

TEST(Rules, WitnessZeroKMeansNoArray) {
  std::mt19937_64 rng(78);
  Csr csr = make_random_csr(rng, 500, 10, 0.2, 0.05, 4);
  fpm::RuleParams p;
  p.max_antecedents = 2;
  p.k_witnesses = 0;
  auto rules = run_rules(csr, 3, 3, p);
  EXPECT_TRUE(rules.witnesses.empty());
  EXPECT_EQ(rules.k_witnesses, 0);
}

TEST(Rules, EmptyInputs) {
  Csr csr{{0, 1, 2}, {1, 2}};
  fpm::RuleParams p;
  p.max_antecedents = 2;
  auto rules = run_rules(csr, 1, 3, p);  // no pairs frequent
  EXPECT_EQ(rules.size(), 0u);
  ASSERT_FALSE(rules.ant_offsets.empty());
  EXPECT_EQ(rules.ant_offsets[0], 0u);
}

TEST(Rules, InvalidParamsThrow) {
  Csr csr{{0, 2, 4}, {1, 2, 1, 3}};
  auto idx = fpm::VerticalIndex::build(csr.offsets.data(), csr.offsets.size(), csr.keys.data(),
                                       csr.keys.size(), 1, 1);
  auto fi = fpm::mine_itemsets(idx, 3, 1);
  fpm::RuleParams p;
  p.max_antecedents = 0;
  EXPECT_THROW(fpm::generate_rules(idx, fi, p), std::invalid_argument);
  p.max_antecedents = 1;
  p.min_confidence = 1.5;
  EXPECT_THROW(fpm::generate_rules(idx, fi, p), std::invalid_argument);
  p.min_confidence = 0.0;
  p.thr.conf = {0.0, 1.2};
  EXPECT_THROW(fpm::generate_rules(idx, fi, p), std::invalid_argument);
  p.thr.conf = {};
  p.thr.lift = {-0.5};
  EXPECT_THROW(fpm::generate_rules(idx, fi, p), std::invalid_argument);
  p.thr.lift = {};
  p.k_witnesses = -1;
  EXPECT_THROW(fpm::generate_rules(idx, fi, p), std::invalid_argument);
}

TEST(Rules, NBasketsMatchesItemsetCounts) {
  std::mt19937_64 rng(91);
  Csr csr = make_random_csr(rng, 800, 12, 0.2, 0.06, 5);
  auto sets = oracle_itemsets(csr.offsets, csr.keys, 4, 3);
  fpm::RuleParams p;
  p.max_antecedents = 2;
  auto rules = run_rules(csr, 4, 3, p);
  for (size_t r = 0; r < rules.size(); ++r) {
    std::vector<uint64_t> z(rules.ant_keys.begin() + rules.ant_offsets[r],
                            rules.ant_keys.begin() + rules.ant_offsets[r + 1]);
    z.push_back(rules.consequent[r]);
    std::sort(z.begin(), z.end());
    EXPECT_EQ(rules.n_baskets[r], static_cast<int64_t>(sets.at(z)));
  }
}

// size_key = min(total_size, 5): itemsets longer than 5 must be governed by
// the LAST threshold entry. Verified by targeted presence/absence of rules
// whose confidence/support is known exactly, not via the shared helper.
TEST(Rules, SizeKeyClampAtFive) {
  // 850 tx with {1,2,3,4,5}; 150 tx with {1,2,3,4,6}. N = 1000.
  // count({1,2,3,4}) = 1000; count({1,2,3,4,5}) = 850; count({1,2,3,4,6}) = 150.
  std::vector<uint64_t> offsets{0}, keys;
  for (int i = 0; i < 850; ++i) {
    for (uint64_t it : {1, 2, 3, 4, 5}) keys.push_back(it);
    offsets.push_back(keys.size());
  }
  for (int i = 0; i < 150; ++i) {
    for (uint64_t it : {1, 2, 3, 4, 6}) keys.push_back(it);
    offsets.push_back(keys.size());
  }
  Csr csr{offsets, keys};

  auto has_rule = [](const fpm::Rules& r, std::set<uint64_t> ant, uint64_t cons) {
    for (size_t i = 0; i < r.size(); ++i) {
      std::set<uint64_t> got(r.ant_keys.begin() + r.ant_offsets[i],
                             r.ant_keys.begin() + r.ant_offsets[i + 1]);
      if (got == ant && r.consequent[i] == cons) return true;
    }
    return false;
  };

  fpm::RuleParams p;
  p.max_antecedents = 6;
  p.min_confidence = 0.0;
  p.thr.conf = {0.0, 0.0, 0.0, 0.0, 0.9};  // size_key 5 (i.e. |Z| >= 5): floor 0.9

  auto rules = run_rules(csr, 1, 7, p);
  // {1,2,3,4} -> {5}: conf = 850/1000 = 0.85, |Z| = 5 -> idx 4 -> dropped.
  EXPECT_FALSE(has_rule(rules, {1, 2, 3, 4}, 5));
  // {1,2,3} -> {5}: conf = 850/1000 = 0.85, |Z| = 4 -> idx 3 = 0 -> kept.
  EXPECT_TRUE(has_rule(rules, {1, 2, 3}, 5));

  // Support clamp likewise: supp({1,2,3,4,5}) = 0.85; drop |Z| >= 5 via idx 4.
  p.thr.conf = {};
  p.thr.supp = {0.0, 0.0, 0.0, 0.0, 0.86};
  rules = run_rules(csr, 1, 7, p);
  EXPECT_FALSE(has_rule(rules, {1, 2, 3, 4}, 5));  // supp 0.85 < 0.86, |Z| = 5
  EXPECT_TRUE(has_rule(rules, {1, 2, 3}, 5));      // |Z| = 4 -> unconstrained
  EXPECT_FALSE(has_rule(rules, {1, 2, 3, 4}, 6));  // supp 0.15 < 0.86, |Z| = 5
}

}  // namespace
