#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "bitset.h"

// Vertical mining driver: builds the item -> transaction-bitset index from a
// CSR database and runs the depth-capped Eclat-style DFS over conditional
// pattern bases.

namespace fpm {

struct WordsDeleter {
  void operator()(uint64_t* p) const { free_words(p); }
};
using WordsPtr = std::unique_ptr<uint64_t, WordsDeleter>;

// Vertical index over a CSR transaction database. Retain it across calls:
// rule witnesses re-intersect its rows.
class VerticalIndex {
 public:
  VerticalIndex() = default;
  VerticalIndex(VerticalIndex&&) = default;
  VerticalIndex& operator=(VerticalIndex&&) = default;
  VerticalIndex(const VerticalIndex&) = delete;
  VerticalIndex& operator=(const VerticalIndex&) = delete;

  // offsets: CSR row pointers, length n_transactions + 1.
  // keys: item ids, length must equal offsets[n_transactions].
  // Only items with support >= min_support_abs get a bitset row.
  // threads: 0 = runtime default (honors OMP_NUM_THREADS), > 0 overrides.
  // Throws std::invalid_argument on malformed CSR / unsupported sizes,
  // std::bad_alloc on allocation failure.
  static VerticalIndex build(const uint64_t* offsets, size_t offsets_len, const uint64_t* keys,
                             size_t keys_len, int64_t min_support_abs, int threads);

  uint64_t n_transactions() const { return n_; }
  size_t words() const { return words_; }  // ceil(N / 64)
  size_t padded_words() const { return pw_; }
  size_t num_items() const { return item_ids_.size(); }  // frequent items only
  int64_t min_support() const { return min_support_; }

  const uint64_t* row(size_t dense) const { return rows_.get() + dense * pw_; }
  uint64_t item_id(size_t dense) const { return item_ids_[dense]; }
  uint64_t item_count(size_t dense) const { return item_counts_[dense]; }

  // Dense index of an item id, or -1 when the item is not frequent.
  int64_t dense_of(uint64_t id) const {
    const auto it = dense_of_.find(id);
    return it == dense_of_.end() ? -1 : static_cast<int64_t>(it->second);
  }

 private:
  WordsPtr rows_;  // num_items * pw_ words, 64-byte aligned, zero padding
  size_t words_ = 0;
  size_t pw_ = 0;
  uint64_t n_ = 0;
  int64_t min_support_ = 0;
  std::vector<uint64_t> item_ids_;     // dense -> original id, ordered by (count, id)
  std::vector<uint64_t> item_counts_;  // exact popcount of each row
  std::unordered_map<uint64_t, uint32_t> dense_of_;
};

struct FrequentItemsets {
  uint64_t n_transactions = 0;
  std::vector<uint64_t> ids;      // itemsets flattened (original item ids)
  std::vector<uint64_t> offsets;  // size = num_itemsets + 1
  std::vector<uint64_t> counts;   // absolute support per itemset
};

// Depth-capped DFS. `db` must have been built with the same min_support_abs
// the caller mines with. Itemsets are emitted in DFS order (deterministic
// regardless of thread count). When debug_store_at_terminal is set, terminal
// levels materialize child bitsets instead of using the store-skip kernel —
// results must be identical; testing hook only.
FrequentItemsets mine_itemsets(const VerticalIndex& db, int max_len, int threads,
                               bool debug_store_at_terminal = false);

// 0 -> OpenMP runtime default (honors OMP_NUM_THREADS), > 0 -> that value.
int effective_threads(int requested);

}  // namespace fpm
