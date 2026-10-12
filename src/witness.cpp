#include "witness.h"

#include <omp.h>
#include <vector>

namespace fpm {

void fill_witnesses(const VerticalIndex& db, Rules& rules, int k, int threads) {
  const size_t num_rules = rules.size();
  const size_t words = db.words();
  rules.witnesses.assign(num_rules * static_cast<size_t>(k), -1);
  if (num_rules == 0 || k == 0) return;

  const int nt = effective_threads(threads);
#pragma omp parallel num_threads(nt)
  {
    std::vector<const uint64_t*> rows;
#pragma omp for schedule(static) nowait
    for (int64_t r = 0; r < static_cast<int64_t>(num_rules); ++r) {
      rows.clear();
      for (uint64_t j = rules.ant_offsets[r]; j < rules.ant_offsets[r + 1]; ++j)
        rows.push_back(db.row(static_cast<size_t>(db.dense_of(rules.ant_keys[j]))));
      rows.push_back(db.row(static_cast<size_t>(db.dense_of(rules.consequent[r]))));

      int32_t* out = rules.witnesses.data() + static_cast<size_t>(r) * static_cast<size_t>(k);
      int filled = 0;
      for (size_t w = 0; w < words && filled < k; ++w) {
        uint64_t v = rows[0][w];
        for (size_t j = 1; j < rows.size() && v; ++j) v &= rows[j][w];
        while (v && filled < k) {
          out[filled++] = static_cast<int32_t>(w * 64 + static_cast<size_t>(__builtin_ctzll(v)));
          v &= v - 1;
        }
      }
    }
  }
}

}  // namespace fpm
