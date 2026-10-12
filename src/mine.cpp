#include "mine.h"

#include <algorithm>
#include <limits>
#include <new>
#include <omp.h>
#include <stdexcept>

namespace fpm {

int effective_threads(int requested) {
  if (requested > 0) return requested;
  const int n = omp_get_max_threads();
  return n > 0 ? n : 1;
}

namespace {

// Parallelize a batch loop only when item_count * words reaches this value;
// below it fork-join overhead exceeds the win (bandwidth-bound kernel).
constexpr int64_t kParallelThresholdWords = 1 << 18;
// Per-level child-slab budget: keeps DFS scratch memory bounded.
constexpr size_t kSlabBudgetBytes = size_t(64) << 20;
// Below this row size, non-temporal stores lose to cached stores.
constexpr size_t kNtMinBytes = size_t(1) << 20;
constexpr size_t kMaxBatch = 4096;

void validate_csr(const uint64_t* offsets, size_t offsets_len, const uint64_t* keys,
                  size_t keys_len) {
  if (offsets == nullptr) throw std::invalid_argument("null offsets buffer");
  if (keys == nullptr && keys_len != 0) throw std::invalid_argument("null keys buffer");
  if (offsets_len == 0) throw std::invalid_argument("offsets must have at least one element");
  if (offsets[0] != 0) throw std::invalid_argument("offsets[0] must be 0");
  for (size_t i = 1; i < offsets_len; ++i)
    if (offsets[i] < offsets[i - 1]) throw std::invalid_argument("offsets not monotone");
  if (offsets[offsets_len - 1] != keys_len)
    throw std::invalid_argument("offsets[N] must equal keys length");
}

}  // namespace

VerticalIndex VerticalIndex::build(const uint64_t* offsets, size_t offsets_len,
                                   const uint64_t* keys, size_t keys_len,
                                   int64_t min_support_abs, int threads) {
  validate_csr(offsets, offsets_len, keys, keys_len);
  if (min_support_abs < 1) throw std::invalid_argument("min_support_abs must be >= 1");
  const uint64_t n = offsets_len - 1;
  if (n > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
    throw std::invalid_argument("too many transactions (witness indices are int32)");

  VerticalIndex idx;
  idx.n_ = n;
  idx.min_support_ = min_support_abs;
  idx.words_ = static_cast<size_t>((n + 63) / 64);
  idx.pw_ = fpm::padded_words(idx.words_);
  const int nt = effective_threads(threads);

  // Pass 1: raw per-item occurrence counts (may overcount duplicate items in
  // one transaction; exact counts are recomputed from the bitsets below).
  std::unordered_map<uint64_t, uint64_t> counts;
  {
    std::vector<std::unordered_map<uint64_t, uint64_t>> local(static_cast<size_t>(nt));
    const size_t chunk = (keys_len + static_cast<size_t>(nt) - 1) / static_cast<size_t>(nt);
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int64_t t = 0; t < nt; ++t) {
      const size_t lo = static_cast<size_t>(t) * chunk;
      const size_t hi = std::min(keys_len, lo + chunk);
      auto& m = local[static_cast<size_t>(t)];
      for (size_t i = lo; i < hi; ++i) ++m[keys[i]];
    }
    for (auto& m : local)
      for (const auto& kv : m) counts[kv.first] += kv.second;
  }

  // Frequent-by-raw-count items, ordered by (count, id) ascending: rare items
  // first, so each node's candidate suffix is as short as useful.
  std::vector<std::pair<uint64_t, uint64_t>> freq;  // (id, raw count)
  for (const auto& kv : counts)
    if (kv.second >= static_cast<uint64_t>(min_support_abs)) freq.push_back(kv);
  std::sort(freq.begin(), freq.end(),
            [](const auto& a, const auto& b) {
              return a.second != b.second ? a.second < b.second : a.first < b.first;
            });
  if (freq.size() > std::numeric_limits<uint32_t>::max())
    throw std::length_error("too many frequent items");

  const size_t m = freq.size();
  idx.item_ids_.reserve(m);
  for (size_t i = 0; i < m; ++i) {
    idx.item_ids_.push_back(freq[i].first);
    idx.dense_of_.emplace(freq[i].first, static_cast<uint32_t>(i));
  }

  const size_t total_words = m * idx.pw_;
  if (m != 0 && total_words / m != idx.pw_) throw std::bad_alloc();
  idx.rows_ = WordsPtr(total_words ? alloc_words(total_words) : nullptr);
  if (total_words && !idx.rows_) throw std::bad_alloc();

  // Pass 2: set bits. Each thread owns a 64-transaction-aligned range, so the
  // touched words are disjoint across threads (no atomics needed).
  if (n > 0 && m > 0) {
    const uint64_t span = (n + static_cast<uint64_t>(nt) - 1) / static_cast<uint64_t>(nt);
    const uint64_t len = std::max<uint64_t>(64, (span + 63) & ~uint64_t(63));
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int64_t c = 0; c < nt; ++c) {
      const uint64_t lo = static_cast<uint64_t>(c) * len;
      const uint64_t hi = std::min<uint64_t>(n, lo + len);
      if (lo >= hi) continue;
      for (uint64_t tx = lo; tx < hi; ++tx) {
        for (uint64_t k = offsets[tx]; k < offsets[tx + 1]; ++k) {
          const auto it = idx.dense_of_.find(keys[k]);
          if (it != idx.dense_of_.end())
            idx.rows_.get()[static_cast<size_t>(it->second) * idx.pw_ +
                            static_cast<size_t>(tx >> 6)] |= uint64_t(1) << (tx & 63);
        }
      }
    }
  }

  // Exact single-item counts (authoritative; raw pass-1 counts can overcount
  // when a transaction repeats an item).
  idx.item_counts_.resize(m);
#pragma omp parallel for schedule(static) num_threads(nt) \
    if(static_cast<int64_t>(m) * static_cast<int64_t>(idx.words_) >= kParallelThresholdWords)
  for (int64_t i = 0; i < static_cast<int64_t>(m); ++i)
    idx.item_counts_[static_cast<size_t>(i)] =
        popcount_words(idx.row(static_cast<size_t>(i)), idx.words_);

  return idx;
}

namespace {

struct MineCtx {
  MineCtx(const VerticalIndex& db_, int64_t minsup_, int max_len_, int nt_, bool debug_store_)
      : db(db_),
        minsup(minsup_),
        max_len(max_len_),
        nt(nt_),
        debug_store_at_terminal(debug_store_) {}

  const VerticalIndex& db;
  int64_t minsup;
  int max_len;
  int nt;
  bool debug_store_at_terminal;

  std::vector<uint32_t> prefix;  // dense ids on the current DFS path
  FrequentItemsets out;

  struct Level {
    WordsPtr slab;  // batch * padded_words child slots
    std::vector<uint64_t> counts;
    size_t batch = 0;
  };
  std::vector<Level> levels;  // index = depth; [0] unused

  void emit(size_t item, uint64_t count) {
    for (uint32_t p : prefix) out.ids.push_back(db.item_id(p));
    out.ids.push_back(db.item_id(item));
    out.counts.push_back(count);
    out.offsets.push_back(out.ids.size());
  }
};

void mine_node(MineCtx& ctx, const uint64_t* parent, size_t lo, int depth) {
  const VerticalIndex& db = ctx.db;
  const size_t words = db.words();
  const size_t pw = db.padded_words();
  const size_t m = db.num_items();
  MineCtx::Level& lv = ctx.levels[static_cast<size_t>(depth)];

  if (lv.batch == 0) {  // lazy per-level slab allocation
    size_t batch = kSlabBudgetBytes / (pw * 8);
    batch = std::max<size_t>(1, std::min({batch, m, kMaxBatch}));
    lv.batch = batch;
    lv.slab = WordsPtr(alloc_words(batch * pw));
    if (!lv.slab) throw std::bad_alloc();
    lv.counts.assign(batch, 0);
  }

  const bool terminal = (depth + 1 == ctx.max_len);
  const bool use_nt = pw * 8 >= kNtMinBytes;
  const uint64_t minsup = static_cast<uint64_t>(ctx.minsup);

  for (size_t b0 = lo; b0 < m; b0 += lv.batch) {
    const size_t n = std::min(lv.batch, m - b0);
    uint64_t* slab = lv.slab.get();
    uint64_t* cnt = lv.counts.data();

    const bool par =
        static_cast<int64_t>(n) * static_cast<int64_t>(words) >= kParallelThresholdWords;
#pragma omp parallel for schedule(dynamic, 64) num_threads(ctx.nt) if(par)
    for (int64_t j = 0; j < static_cast<int64_t>(n); ++j) {
      const uint64_t* r = db.row(b0 + static_cast<size_t>(j));
      if (terminal && !ctx.debug_store_at_terminal)
        cnt[j] = intersect_count_nostore(parent, r, words);
      else if (!terminal && use_nt)
        cnt[j] = intersect_count_nt(parent, r, slab + static_cast<size_t>(j) * pw, words);
      else
        cnt[j] = intersect_count(parent, r, slab + static_cast<size_t>(j) * pw, words);
    }

    for (size_t j = 0; j < n; ++j) {
      if (cnt[j] >= minsup) {
        const size_t item = b0 + j;
        ctx.emit(item, cnt[j]);
        if (!terminal && item + 1 < m) {
          ctx.prefix.push_back(static_cast<uint32_t>(item));
          mine_node(ctx, slab + j * pw, item + 1, depth + 1);
          ctx.prefix.pop_back();
        }
      }
    }
  }
}

}  // namespace

FrequentItemsets mine_itemsets(const VerticalIndex& db, int max_len, int threads,
                               bool debug_store_at_terminal) {
  if (max_len < 1) throw std::invalid_argument("max_len must be >= 1");
  MineCtx ctx{db, db.min_support(), max_len, effective_threads(threads),
              debug_store_at_terminal};
  ctx.levels.resize(static_cast<size_t>(max_len));
  ctx.out.n_transactions = db.n_transactions();
  ctx.out.offsets.push_back(0);

  const size_t m = db.num_items();
  for (size_t item = 0; item < m; ++item) {
    const uint64_t c = db.item_count(item);
    if (c < static_cast<uint64_t>(ctx.minsup)) continue;
    ctx.emit(item, c);
    if (max_len > 1 && item + 1 < m) {
      ctx.prefix.push_back(static_cast<uint32_t>(item));
      mine_node(ctx, db.row(item), item + 1, 1);
      ctx.prefix.pop_back();
    }
  }
  return std::move(ctx.out);
}

}  // namespace fpm
