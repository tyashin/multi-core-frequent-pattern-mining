#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "bitset.h"

namespace {

// Independent naive oracle: per-bit loop, shares no code with the kernels.
uint64_t naive_popcount(const uint64_t* a, size_t words) {
  uint64_t c = 0;
  for (size_t i = 0; i < words; ++i) {
    uint64_t w = a[i];
    while (w) {
      c += w & 1;
      w >>= 1;
    }
  }
  return c;
}

uint64_t naive_intersect_count(const uint64_t* a, const uint64_t* b, uint64_t* out,
                               size_t words) {
  uint64_t c = 0;
  for (size_t i = 0; i < words; ++i) {
    const uint64_t v = a[i] & b[i];
    if (out) out[i] = v;
    uint64_t w = v;
    while (w) {
      c += w & 1;
      w >>= 1;
    }
  }
  return c;
}

struct Buffer {
  uint64_t* p = nullptr;
  size_t words = 0;
  Buffer() = default;
  explicit Buffer(size_t w) : p(fpm::alloc_words(w)), words(w) {}
  ~Buffer() { fpm::free_words(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void fill_random(std::mt19937_64& rng) {
    for (size_t i = 0; i < words; ++i) p[i] = rng();
  }
};

std::vector<fpm::Isa> supported_isas() {
  std::vector<fpm::Isa> v{fpm::Isa::kScalar};
  const fpm::Isa best = fpm::detect_isa();
  if (best >= fpm::Isa::kAvx2) v.push_back(fpm::Isa::kAvx2);
  if (best >= fpm::Isa::kAvx512) v.push_back(fpm::Isa::kAvx512);
  return v;
}

using PopFn = uint64_t (*)(const uint64_t*, size_t);
using IcFn = uint64_t (*)(const uint64_t*, const uint64_t*, uint64_t*, size_t);
using IcNsFn = uint64_t (*)(const uint64_t*, const uint64_t*, size_t);

PopFn pop_kernel(fpm::Isa isa) {
  switch (isa) {
    case fpm::Isa::kScalar: return fpm::kernels::popcount_scalar;
    case fpm::Isa::kAvx2: return fpm::kernels::popcount_avx2;
    case fpm::Isa::kAvx512: return fpm::kernels::popcount_avx512;
  }
  return nullptr;
}

IcFn ic_kernel(fpm::Isa isa) {
  switch (isa) {
    case fpm::Isa::kScalar: return fpm::kernels::intersect_count_scalar;
    case fpm::Isa::kAvx2: return fpm::kernels::intersect_count_avx2;
    case fpm::Isa::kAvx512: return fpm::kernels::intersect_count_avx512;
  }
  return nullptr;
}

IcNsFn ic_nostore_kernel(fpm::Isa isa) {
  switch (isa) {
    case fpm::Isa::kScalar: return fpm::kernels::intersect_count_nostore_scalar;
    case fpm::Isa::kAvx2: return fpm::kernels::intersect_count_nostore_avx2;
    case fpm::Isa::kAvx512: return fpm::kernels::intersect_count_nostore_avx512;
  }
  return nullptr;
}

const std::vector<size_t>& test_sizes() {
  static const std::vector<size_t> sizes = {0,  1,  2,  3,   4,   5,   7,   8,   9,
                                            15, 16, 17, 31,  32,  33,  63,  64,  65,
                                            124, 125, 127, 1000, 1024, 4096};
  return sizes;
}

TEST(BitsetAlloc, AlignedAndZeroed) {
  Buffer buf(fpm::padded_words(100));
  ASSERT_NE(buf.p, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(buf.p) % 64, 0u);
  for (size_t i = 0; i < buf.words; ++i) EXPECT_EQ(buf.p[i], 0u);
}

TEST(BitsetAlloc, NullForZeroWords) {
  EXPECT_EQ(fpm::alloc_words(0), nullptr);
  fpm::free_words(nullptr);  // must be safe
}

TEST(BitsetPopcount, MatchesNaiveAllIsas) {
  std::mt19937_64 rng(42);
  for (size_t words : test_sizes()) {
    Buffer buf(words ? words : 1);  // alloc 1 word for the words=0 case
    buf.fill_random(rng);
    const uint64_t ref = naive_popcount(buf.p, words);
    for (fpm::Isa isa : supported_isas())
      EXPECT_EQ(pop_kernel(isa)(buf.p, words), ref)
          << "isa=" << fpm::isa_name(isa) << " words=" << words;
  }
}

TEST(BitsetPopcount, EdgePatterns) {
  for (size_t words : {size_t(1), size_t(8), size_t(64)}) {
    Buffer buf(words);
    for (fpm::Isa isa : supported_isas()) {
      EXPECT_EQ(pop_kernel(isa)(buf.p, words), 0u);  // all zeros
    }
    for (size_t i = 0; i < words; ++i) buf.p[i] = ~uint64_t(0);
    for (fpm::Isa isa : supported_isas()) {
      EXPECT_EQ(pop_kernel(isa)(buf.p, words), words * 64);
    }
  }
}

TEST(BitsetIntersect, DisjointIdenticalEmpty) {
  std::mt19937_64 rng(7);
  for (size_t words : test_sizes()) {
    if (words == 0) continue;
    Buffer a(words), b(words), out(words);
    a.fill_random(rng);
    b.fill_random(rng);

    // Disjoint: mask b to the complement of a.
    for (size_t i = 0; i < words; ++i) b.p[i] = ~a.p[i];
    for (fpm::Isa isa : supported_isas()) {
      EXPECT_EQ(ic_kernel(isa)(a.p, b.p, out.p, words), 0u)
          << "disjoint isa=" << fpm::isa_name(isa);
      for (size_t i = 0; i < words; ++i) EXPECT_EQ(out.p[i], 0u);
    }

    // Identical: intersection is the input itself.
    for (fpm::Isa isa : supported_isas()) {
      EXPECT_EQ(ic_kernel(isa)(a.p, a.p, out.p, words), naive_popcount(a.p, words))
          << "identical isa=" << fpm::isa_name(isa);
      for (size_t i = 0; i < words; ++i) EXPECT_EQ(out.p[i], a.p[i]);
    }
  }

  // Empty range: count 0, no writes.
  Buffer a(1), b(1), out(1);
  a.p[0] = ~uint64_t(0);
  b.p[0] = ~uint64_t(0);
  out.p[0] = 0xdeadbeefcafebabe;
  for (fpm::Isa isa : supported_isas()) {
    EXPECT_EQ(ic_kernel(isa)(a.p, b.p, out.p, 0), 0u);
    EXPECT_EQ(ic_nostore_kernel(isa)(a.p, b.p, 0), 0u);
  }
  EXPECT_EQ(out.p[0], 0xdeadbeefcafebabe);
}

TEST(BitsetIntersect, RandomMatchesNaiveAllVariants) {
  std::mt19937_64 rng(1234);
  for (size_t words : test_sizes()) {
    if (words == 0) continue;
    Buffer a(words), b(words), out(words), ref_out(words);
    a.fill_random(rng);
    b.fill_random(rng);
    const uint64_t ref = naive_intersect_count(a.p, b.p, ref_out.p, words);

    for (fpm::Isa isa : supported_isas()) {
      for (size_t i = 0; i < words; ++i) out.p[i] = ~uint64_t(0);  // poison
      EXPECT_EQ(ic_kernel(isa)(a.p, b.p, out.p, words), ref)
          << "store isa=" << fpm::isa_name(isa) << " words=" << words;
      for (size_t i = 0; i < words; ++i) EXPECT_EQ(out.p[i], ref_out.p[i]);

      EXPECT_EQ(ic_nostore_kernel(isa)(a.p, b.p, words), ref)
          << "nostore isa=" << fpm::isa_name(isa) << " words=" << words;
    }
  }
}

TEST(BitsetIntersect, NonTemporalMatchesRegular) {
  std::mt19937_64 rng(99);
  for (size_t words : {size_t(8), size_t(33), size_t(1024), size_t(8192)}) {
    Buffer a(fpm::padded_words(words)), b(fpm::padded_words(words));
    Buffer out(fpm::padded_words(words)), ref(fpm::padded_words(words));
    a.fill_random(rng);
    b.fill_random(rng);
    const uint64_t ref_count =
        fpm::kernels::intersect_count_scalar(a.p, b.p, ref.p, words);

    const fpm::Isa best = fpm::detect_isa();
    if (best >= fpm::Isa::kAvx2) {
      EXPECT_EQ(fpm::kernels::intersect_count_nt_avx2(a.p, b.p, out.p, words), ref_count);
      for (size_t i = 0; i < words; ++i) EXPECT_EQ(out.p[i], ref.p[i]);
    }
    if (best >= fpm::Isa::kAvx512) {
      EXPECT_EQ(fpm::kernels::intersect_count_nt_avx512(a.p, b.p, out.p, words), ref_count);
      for (size_t i = 0; i < words; ++i) EXPECT_EQ(out.p[i], ref.p[i]);
    }
  }
}

TEST(BitsetDispatch, ForcedIsasBitwiseIdentical) {
  std::mt19937_64 rng(555);
  constexpr size_t words = 777;
  Buffer a(fpm::padded_words(words)), b(fpm::padded_words(words));
  Buffer ref(fpm::padded_words(words)), out(fpm::padded_words(words));
  a.fill_random(rng);
  b.fill_random(rng);

  fpm::force_isa(fpm::Isa::kScalar);
  const uint64_t ref_pop = fpm::popcount_words(a.p, words);
  const uint64_t ref_ic = fpm::intersect_count(a.p, b.p, ref.p, words);
  const uint64_t ref_ns = fpm::intersect_count_nostore(a.p, b.p, words);

  for (fpm::Isa isa : supported_isas()) {
    fpm::force_isa(isa);
    EXPECT_EQ(fpm::popcount_words(a.p, words), ref_pop);
    for (size_t i = 0; i < words; ++i) out.p[i] = ~uint64_t(0);
    EXPECT_EQ(fpm::intersect_count(a.p, b.p, out.p, words), ref_ic);
    for (size_t i = 0; i < words; ++i) EXPECT_EQ(out.p[i], ref.p[i]);
    EXPECT_EQ(fpm::intersect_count_nostore(a.p, b.p, words), ref_ns);
    Buffer nt_out(fpm::padded_words(words));
    EXPECT_EQ(fpm::intersect_count_nt(a.p, b.p, nt_out.p, words), ref_ic);
    for (size_t i = 0; i < words; ++i) EXPECT_EQ(nt_out.p[i], ref.p[i]);
  }
  fpm::clear_forced_isa();
}

TEST(BitsetDispatch, ForceUnsupportedClamps) {
  fpm::force_isa(fpm::Isa::kAvx512);
  const fpm::Isa cur = fpm::current_isa();
  EXPECT_LE(static_cast<int>(cur), static_cast<int>(fpm::detect_isa()));
  if (fpm::detect_isa() < fpm::Isa::kAvx512) {
    EXPECT_NE(cur, fpm::Isa::kAvx512);
  }
  // Dispatch still works after clamping.
  Buffer a(8);
  a.p[0] = 0xff;
  EXPECT_EQ(fpm::popcount_words(a.p, 8), 8u);
  fpm::clear_forced_isa();
}

TEST(BitsetDispatch, ParseIsaName) {
  fpm::Isa isa;
  EXPECT_TRUE(fpm::parse_isa_name("scalar", isa));
  EXPECT_EQ(isa, fpm::Isa::kScalar);
  EXPECT_TRUE(fpm::parse_isa_name("avx2", isa));
  EXPECT_EQ(isa, fpm::Isa::kAvx2);
  EXPECT_TRUE(fpm::parse_isa_name("avx512", isa));
  EXPECT_EQ(isa, fpm::Isa::kAvx512);
  EXPECT_FALSE(fpm::parse_isa_name("avx1024", isa));
  EXPECT_FALSE(fpm::parse_isa_name("", isa));
  EXPECT_FALSE(fpm::parse_isa_name(nullptr, isa));
}

TEST(BitsetPaddedWords, RoundsTo64ByteStride) {
  EXPECT_EQ(fpm::padded_words(0), 0u);
  EXPECT_EQ(fpm::padded_words(1), 8u);
  EXPECT_EQ(fpm::padded_words(7), 8u);
  EXPECT_EQ(fpm::padded_words(8), 8u);
  EXPECT_EQ(fpm::padded_words(9), 16u);
  EXPECT_EQ(fpm::padded_words(15625), 15632u);
}

}  // namespace
