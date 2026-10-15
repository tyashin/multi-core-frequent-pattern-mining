#pragma once

#include <cstdint>
#include <vector>

#include "mine.h"

// Association-rule expansion from mined frequent itemsets: single-consequent
// rules X -> {c} with support / confidence / lift computed from integer
// counts (fractions derived only at emission, no rounding drift).

namespace fpm {

// Per-length thresholds indexed by size_key = total_size, where
// total_size = |antecedent| + 1 = |Z|. Entry index = size_key - 1;
// entries missing from the array mean 0.0 (no constraint).
struct Thresholds {
  std::vector<double> supp;  // fraction in [0, 1]
  std::vector<double> conf;  // fraction in [0, 1]
  std::vector<double> lift;  // >= 0

  double supp_at(int size_key) const;
  double conf_at(int size_key) const;
  double lift_at(int size_key) const;
};

struct RuleParams {
  int max_antecedents = 1;    // A: only rules with |antecedent| <= A
  double min_confidence = 0.0;  // global generation-time floor (Spark semantics)
  Thresholds thr;
  int threads = 0;            // 0 = runtime default
  int k_witnesses = 0;        // witness transactions per rule; 0 = skip pass
};

struct Rules {
  std::vector<uint64_t> ant_offsets;  // R + 1
  std::vector<uint64_t> ant_keys;     // antecedent item ids (CSR)
  std::vector<uint64_t> consequent;   // R
  std::vector<double> support;        // count(Z) / N
  std::vector<double> confidence;     // count(Z) / count(X)
  std::vector<double> lift;           // confidence / (count({c}) / N)
  std::vector<int64_t> n_baskets;     // exact union counts = count(Z)
  std::vector<int32_t> witnesses;     // R * k_witnesses, -1 padded
  int k_witnesses = 0;

  size_t size() const { return consequent.size(); }
};

// `fi` must come from mine_itemsets on `db`. Rules are emitted in mining
// order of Z, consequents in itemset order — deterministic for any
// fixed thread count, and bitwise-identical across thread counts.
Rules generate_rules(const VerticalIndex& db, const FrequentItemsets& fi,
                     const RuleParams& params);

// Full pipeline: build index, mine itemsets, expand rules. One call in,
// one call out. max_len is the itemset depth cap (e.g. A + 1);
// params.max_antecedents is the rule-side cap A.
Rules mine_rules(const uint64_t* offsets, size_t offsets_len, const uint64_t* keys,
                 size_t keys_len, int64_t min_support_abs, int max_len,
                 const RuleParams& params);

}  // namespace fpm
