#include "rules.h"

#include <algorithm>
#include <omp.h>
#include <stdexcept>
#include <unordered_map>

#include "witness.h"

namespace fpm {

namespace {

double thr_at(const std::vector<double>& a, int size_key) {
  const int idx = size_key - 1;
  return (idx >= 0 && static_cast<size_t>(idx) < a.size()) ? a[static_cast<size_t>(idx)] : 0.0;
}

}  // namespace

double Thresholds::supp_at(int size_key) const { return thr_at(supp, size_key); }
double Thresholds::conf_at(int size_key) const { return thr_at(conf, size_key); }
double Thresholds::lift_at(int size_key) const { return thr_at(lift, size_key); }

namespace {

void validate_params(const RuleParams& p) {
  if (p.max_antecedents < 1) throw std::invalid_argument("max_antecedents must be >= 1");
  if (p.min_confidence < 0.0 || p.min_confidence > 1.0)
    throw std::invalid_argument("min_confidence must be in [0, 1]");
  if (p.k_witnesses < 0) throw std::invalid_argument("k_witnesses must be >= 0");
  for (double v : p.thr.supp)
    if (v < 0.0 || v > 1.0) throw std::invalid_argument("supp_thr entries must be in [0, 1]");
  for (double v : p.thr.conf)
    if (v < 0.0 || v > 1.0) throw std::invalid_argument("conf_thr entries must be in [0, 1]");
  for (double v : p.thr.lift)
    if (v < 0.0) throw std::invalid_argument("lift_thr entries must be >= 0");
}

struct VecHash {
  size_t operator()(const std::vector<uint32_t>& v) const noexcept {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ v.size();
    for (uint32_t x : v) {
      h ^= 0x9e3779b97f4a7c15ull + static_cast<uint64_t>(x) * 0x100000001b3ull;
      h *= 0xff51afd7ed558ccdull;
      h ^= h >> 33;
    }
    return static_cast<size_t>(h);
  }
};

using CountMap = std::unordered_map<std::vector<uint32_t>, uint64_t, VecHash>;

CountMap build_count_map(const VerticalIndex& db, const FrequentItemsets& fi) {
  CountMap m;
  m.reserve(fi.counts.size() * 2);
  std::vector<uint32_t> key;
  for (size_t i = 0; i + 1 < fi.offsets.size(); ++i) {
    key.clear();
    for (uint64_t j = fi.offsets[i]; j < fi.offsets[i + 1]; ++j) {
      const int64_t d = db.dense_of(fi.ids[j]);
      // Every emitted itemset's items are frequent by anti-monotonicity.
      if (d < 0) throw std::logic_error("mined item id missing from index");
      key.push_back(static_cast<uint32_t>(d));
    }
    // Mining emits dense-ascending order, so `key` is already canonical.
    m.emplace(key, fi.counts[i]);
  }
  return m;
}

uint64_t count_of(const CountMap& m, const std::vector<uint32_t>& key) {
  const auto it = m.find(key);
  if (it == m.end()) throw std::logic_error("frequent subset missing from count map");
  return it->second;
}

}  // namespace

Rules generate_rules(const VerticalIndex& db, const FrequentItemsets& fi,
                     const RuleParams& params) {
  validate_params(params);
  const int nt = effective_threads(params.threads);
  const uint64_t n_tx = fi.n_transactions;
  const size_t num_sets = fi.counts.size();

  const CountMap counts = build_count_map(db, fi);

  // Per-chunk partial results, merged in chunk order: output order depends
  // only on itemset order, not on thread scheduling.
  std::vector<Rules> parts(static_cast<size_t>(nt));
  const size_t chunk = (num_sets + static_cast<size_t>(nt) - 1) / static_cast<size_t>(nt);

#pragma omp parallel for schedule(static) num_threads(nt)
  for (int64_t t = 0; t < nt; ++t) {
    const size_t lo = static_cast<size_t>(t) * chunk;
    const size_t hi = std::min(num_sets, lo + chunk);
    Rules& out = parts[static_cast<size_t>(t)];
    out.ant_offsets.push_back(0);
    std::vector<uint32_t> zdense, xdense;

    for (size_t i = lo; i < hi; ++i) {
      const uint64_t zb = fi.offsets[i];
      const uint64_t ze = fi.offsets[i + 1];
      const size_t zsize = ze - zb;
      if (zsize < 2) continue;
      if (static_cast<int>(zsize) - 1 > params.max_antecedents) continue;

      const uint64_t cnt_z = fi.counts[i];
      const int size_key = std::min<int>(static_cast<int>(zsize), Thresholds::kSizeKeyMax);
      const double supp = static_cast<double>(cnt_z) / static_cast<double>(n_tx);
      if (supp < params.thr.supp_at(size_key)) continue;

      zdense.clear();
      for (uint64_t j = zb; j < ze; ++j)
        zdense.push_back(static_cast<uint32_t>(db.dense_of(fi.ids[j])));

      for (size_t p = 0; p < zsize; ++p) {
        xdense.clear();
        for (size_t j = 0; j < zsize; ++j)
          if (j != p) xdense.push_back(zdense[j]);
        const uint64_t cnt_x = count_of(counts, xdense);
        const uint64_t cnt_c = db.item_count(zdense[p]);

        const double conf = static_cast<double>(cnt_z) / static_cast<double>(cnt_x);
        if (conf < std::max(params.min_confidence, params.thr.conf_at(size_key))) continue;
        const double lift = conf * static_cast<double>(n_tx) / static_cast<double>(cnt_c);
        if (lift < params.thr.lift_at(size_key)) continue;

        for (size_t j = 0; j < zsize; ++j)
          if (j != p) out.ant_keys.push_back(db.item_id(zdense[j]));
        out.ant_offsets.push_back(out.ant_keys.size());
        out.consequent.push_back(db.item_id(zdense[p]));
        out.support.push_back(supp);
        out.confidence.push_back(conf);
        out.lift.push_back(lift);
        out.n_baskets.push_back(static_cast<int64_t>(cnt_z));
      }
    }
  }

  Rules result;
  {
    size_t total_rules = 0, total_ant = 0;
    for (const auto& pt : parts) {
      total_rules += pt.size();
      total_ant += pt.ant_keys.size();
    }
    result.ant_offsets.reserve(total_rules + 1);
    result.ant_keys.reserve(total_ant);
    result.consequent.reserve(total_rules);
    result.support.reserve(total_rules);
    result.confidence.reserve(total_rules);
    result.lift.reserve(total_rules);
    result.n_baskets.reserve(total_rules);
    result.ant_offsets.push_back(0);
    for (auto& pt : parts) {
      for (size_t r = 0; r < pt.size(); ++r) {
        result.ant_keys.insert(result.ant_keys.end(),
                               pt.ant_keys.begin() + pt.ant_offsets[r],
                               pt.ant_keys.begin() + pt.ant_offsets[r + 1]);
        result.ant_offsets.push_back(result.ant_keys.size());
      }
      result.consequent.insert(result.consequent.end(), pt.consequent.begin(),
                               pt.consequent.end());
      result.support.insert(result.support.end(), pt.support.begin(), pt.support.end());
      result.confidence.insert(result.confidence.end(), pt.confidence.begin(),
                               pt.confidence.end());
      result.lift.insert(result.lift.end(), pt.lift.begin(), pt.lift.end());
      result.n_baskets.insert(result.n_baskets.end(), pt.n_baskets.begin(),
                              pt.n_baskets.end());
    }
  }
  result.k_witnesses = params.k_witnesses;

  if (params.k_witnesses > 0 && !result.consequent.empty())
    fill_witnesses(db, result, params.k_witnesses, nt);
  return result;
}

}  // namespace fpm
