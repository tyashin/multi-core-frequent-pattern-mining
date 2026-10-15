#include "bitset.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#define FPM_X86 1
#include <immintrin.h>
#else
#define FPM_X86 0
#endif

namespace fpm {
namespace kernels {

// ---------------------------------------------------------------- scalar ---

uint64_t popcount_scalar(const uint64_t* a, size_t words) noexcept {
  uint64_t c = 0;
  for (size_t i = 0; i < words; ++i) c += static_cast<uint64_t>(__builtin_popcountll(a[i]));
  return c;
}

uint64_t intersect_count_scalar(const uint64_t* a, const uint64_t* b, uint64_t* out,
                                size_t words) noexcept {
  uint64_t c = 0;
  for (size_t i = 0; i < words; ++i) {
    const uint64_t v = a[i] & b[i];
    out[i] = v;
    c += static_cast<uint64_t>(__builtin_popcountll(v));
  }
  return c;
}

uint64_t intersect_count_nostore_scalar(const uint64_t* a, const uint64_t* b,
                                        size_t words) noexcept {
  uint64_t c = 0;
  for (size_t i = 0; i < words; ++i)
    c += static_cast<uint64_t>(__builtin_popcountll(a[i] & b[i]));
  return c;
}

#if FPM_X86

// ------------------------------------------------------------------ AVX2 ---
// Muła nibble-LUT popcount ("Faster Population Counts Using AVX2", 2016).
// Byte accumulators hold at most 8 per vector, so flush via SAD every
// 31 vectors (31*8 = 248 <= 255).

__attribute__((target("avx2"))) static inline __m256i avx2_lookup() {
  return _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                          0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
}

__attribute__((target("avx2"))) static inline __m256i popcnt256(__m256i v, __m256i lookup,
                                                                __m256i low_mask) {
  const __m256i lo = _mm256_and_si256(v, low_mask);
  const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
  return _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo), _mm256_shuffle_epi8(lookup, hi));
}

__attribute__((target("avx2"))) static inline uint64_t hsum256(__m256i v) {
  const __m128i s = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  return static_cast<uint64_t>(_mm_cvtsi128_si64(s)) +
         static_cast<uint64_t>(_mm_extract_epi64(s, 1));
}

constexpr size_t kAvx2Block = 31 * 4;  // words per SAD flush

#define FPM_AVX2_TARGET __attribute__((target("avx2")))

FPM_AVX2_TARGET uint64_t popcount_avx2(const uint64_t* a, size_t words) noexcept {
  const __m256i lookup = avx2_lookup();
  const __m256i low_mask = _mm256_set1_epi8(0x0f);
  const size_t vec_words = words & ~size_t(3);
  __m256i total = _mm256_setzero_si256();
  size_t i = 0;
  while (i < vec_words) {
    const size_t block_end = (vec_words - i < kAvx2Block) ? vec_words : i + kAvx2Block;
    __m256i acc = _mm256_setzero_si256();
    for (; i < block_end; i += 4)
      acc = _mm256_add_epi8(
          acc, popcnt256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)), lookup,
                           low_mask));
    total = _mm256_add_epi64(total, _mm256_sad_epu8(acc, _mm256_setzero_si256()));
  }
  uint64_t c = hsum256(total);
  for (; i < words; ++i) c += static_cast<uint64_t>(__builtin_popcountll(a[i]));
  return c;
}

FPM_AVX2_TARGET uint64_t intersect_count_avx2(const uint64_t* a, const uint64_t* b, uint64_t* out,
                              size_t words) noexcept {
  const __m256i lookup = avx2_lookup();
  const __m256i low_mask = _mm256_set1_epi8(0x0f);
  const size_t vec_words = words & ~size_t(3);
  __m256i total = _mm256_setzero_si256();
  size_t i = 0;
  while (i < vec_words) {
    const size_t block_end = (vec_words - i < kAvx2Block) ? vec_words : i + kAvx2Block;
    __m256i acc = _mm256_setzero_si256();
    for (; i < block_end; i += 4) {
      const __m256i v =
          _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)),
                           _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i)));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), v);
      acc = _mm256_add_epi8(acc, popcnt256(v, lookup, low_mask));
    }
    total = _mm256_add_epi64(total, _mm256_sad_epu8(acc, _mm256_setzero_si256()));
  }
  uint64_t c = hsum256(total);
  for (; i < words; ++i) {
    const uint64_t v = a[i] & b[i];
    out[i] = v;
    c += static_cast<uint64_t>(__builtin_popcountll(v));
  }
  return c;
}

FPM_AVX2_TARGET uint64_t intersect_count_nostore_avx2(const uint64_t* a, const uint64_t* b,
                                      size_t words) noexcept {
  const __m256i lookup = avx2_lookup();
  const __m256i low_mask = _mm256_set1_epi8(0x0f);
  const size_t vec_words = words & ~size_t(3);
  __m256i total = _mm256_setzero_si256();
  size_t i = 0;
  while (i < vec_words) {
    const size_t block_end = (vec_words - i < kAvx2Block) ? vec_words : i + kAvx2Block;
    __m256i acc = _mm256_setzero_si256();
    for (; i < block_end; i += 4) {
      const __m256i v =
          _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)),
                           _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i)));
      acc = _mm256_add_epi8(acc, popcnt256(v, lookup, low_mask));
    }
    total = _mm256_add_epi64(total, _mm256_sad_epu8(acc, _mm256_setzero_si256()));
  }
  uint64_t c = hsum256(total);
  for (; i < words; ++i) c += static_cast<uint64_t>(__builtin_popcountll(a[i] & b[i]));
  return c;
}

FPM_AVX2_TARGET uint64_t intersect_count_nt_avx2(const uint64_t* a, const uint64_t* b, uint64_t* out,
                                 size_t words) noexcept {
  const __m256i lookup = avx2_lookup();
  const __m256i low_mask = _mm256_set1_epi8(0x0f);
  const size_t vec_words = words & ~size_t(3);
  __m256i total = _mm256_setzero_si256();
  size_t i = 0;
  while (i < vec_words) {
    const size_t block_end = (vec_words - i < kAvx2Block) ? vec_words : i + kAvx2Block;
    __m256i acc = _mm256_setzero_si256();
    for (; i < block_end; i += 4) {
      const __m256i v =
          _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)),
                           _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i)));
      _mm256_stream_si256(reinterpret_cast<__m256i*>(out + i), v);
      acc = _mm256_add_epi8(acc, popcnt256(v, lookup, low_mask));
    }
    total = _mm256_add_epi64(total, _mm256_sad_epu8(acc, _mm256_setzero_si256()));
  }
  uint64_t c = hsum256(total);
  for (; i < words; ++i) {
    const uint64_t v = a[i] & b[i];
    out[i] = v;
    c += static_cast<uint64_t>(__builtin_popcountll(v));
  }
  _mm_sfence();
  return c;
}

// --------------------------------------------------------------- AVX-512 ---

#define FPM_AVX512_TARGET __attribute__((target("avx512f,avx512vpopcntdq")))

// Lane-sum for __m512i of uint64. _mm512_reduce_add_epi64 goes through
// _mm512_extracti64x4_epi64, whose GCC 12 header implementation trips a
// -Wuninitialized false positive on _mm256_undefined_si256 (fixed in GCC 13)
// — fatal under -Werror. Store-and-sum avoids that header path; it runs once
// per kernel pass, so the cost is negligible.
FPM_AVX512_TARGET static inline uint64_t hsum512(__m512i v) {
  uint64_t buf[8];
  _mm512_storeu_si512(reinterpret_cast<void*>(buf), v);
  uint64_t c = 0;
  for (size_t i = 0; i < 8; ++i) c += buf[i];
  return c;
}

FPM_AVX512_TARGET uint64_t popcount_avx512(const uint64_t* a, size_t words) noexcept {
  __m512i total = _mm512_setzero_si512();
  size_t i = 0;
  for (; i + 8 <= words; i += 8)
    total = _mm512_add_epi64(
        total, _mm512_popcnt_epi64(_mm512_loadu_si512(reinterpret_cast<const void*>(a + i))));
  uint64_t c = hsum512(total);
  if (i < words) {
    const __mmask8 m = static_cast<__mmask8>((1u << (words - i)) - 1u);
    const __m512i v = _mm512_maskz_loadu_epi64(m, reinterpret_cast<const void*>(a + i));
    c += hsum512(_mm512_popcnt_epi64(v));
  }
  return c;
}

FPM_AVX512_TARGET uint64_t intersect_count_avx512(const uint64_t* a, const uint64_t* b, uint64_t* out,
                                size_t words) noexcept {
  __m512i total = _mm512_setzero_si512();
  size_t i = 0;
  for (; i + 8 <= words; i += 8) {
    const __m512i v = _mm512_and_si512(_mm512_loadu_si512(reinterpret_cast<const void*>(a + i)),
                                       _mm512_loadu_si512(reinterpret_cast<const void*>(b + i)));
    _mm512_storeu_si512(reinterpret_cast<void*>(out + i), v);
    total = _mm512_add_epi64(total, _mm512_popcnt_epi64(v));
  }
  uint64_t c = hsum512(total);
  if (i < words) {
    const __mmask8 m = static_cast<__mmask8>((1u << (words - i)) - 1u);
    const __m512i v =
        _mm512_and_si512(_mm512_maskz_loadu_epi64(m, reinterpret_cast<const void*>(a + i)),
                         _mm512_maskz_loadu_epi64(m, reinterpret_cast<const void*>(b + i)));
    _mm512_mask_storeu_epi64(reinterpret_cast<void*>(out + i), m, v);
    c += hsum512(_mm512_popcnt_epi64(v));
  }
  return c;
}

FPM_AVX512_TARGET uint64_t intersect_count_nostore_avx512(const uint64_t* a, const uint64_t* b,
                                        size_t words) noexcept {
  __m512i total = _mm512_setzero_si512();
  size_t i = 0;
  for (; i + 8 <= words; i += 8) {
    const __m512i v = _mm512_and_si512(_mm512_loadu_si512(reinterpret_cast<const void*>(a + i)),
                                       _mm512_loadu_si512(reinterpret_cast<const void*>(b + i)));
    total = _mm512_add_epi64(total, _mm512_popcnt_epi64(v));
  }
  uint64_t c = hsum512(total);
  if (i < words) {
    const __mmask8 m = static_cast<__mmask8>((1u << (words - i)) - 1u);
    const __m512i v =
        _mm512_and_si512(_mm512_maskz_loadu_epi64(m, reinterpret_cast<const void*>(a + i)),
                         _mm512_maskz_loadu_epi64(m, reinterpret_cast<const void*>(b + i)));
    c += hsum512(_mm512_popcnt_epi64(v));
  }
  return c;
}

FPM_AVX512_TARGET uint64_t intersect_count_nt_avx512(const uint64_t* a, const uint64_t* b, uint64_t* out,
                                   size_t words) noexcept {
  __m512i total = _mm512_setzero_si512();
  size_t i = 0;
  for (; i + 8 <= words; i += 8) {
    const __m512i v = _mm512_and_si512(_mm512_loadu_si512(reinterpret_cast<const void*>(a + i)),
                                       _mm512_loadu_si512(reinterpret_cast<const void*>(b + i)));
    _mm512_stream_si512(reinterpret_cast<__m512i*>(out + i), v);
    total = _mm512_add_epi64(total, _mm512_popcnt_epi64(v));
  }
  uint64_t c = hsum512(total);
  if (i < words) {
    const __mmask8 m = static_cast<__mmask8>((1u << (words - i)) - 1u);
    const __m512i v =
        _mm512_and_si512(_mm512_maskz_loadu_epi64(m, reinterpret_cast<const void*>(a + i)),
                         _mm512_maskz_loadu_epi64(m, reinterpret_cast<const void*>(b + i)));
    for (size_t j = i; j < words; ++j) {
      const uint64_t w = a[j] & b[j];
      out[j] = w;
    }
    c += hsum512(_mm512_popcnt_epi64(v));
  }
  _mm_sfence();
  return c;
}

#endif  // FPM_X86

}  // namespace kernels

// ------------------------------------------------------------- ISA dispatch ---

Isa detect_isa() noexcept {
#if FPM_X86
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vpopcntdq"))
    return Isa::kAvx512;
  if (__builtin_cpu_supports("avx2")) return Isa::kAvx2;
#endif
  return Isa::kScalar;
}

const char* isa_name(Isa isa) noexcept {
  switch (isa) {
    case Isa::kScalar: return "scalar";
    case Isa::kAvx2: return "avx2";
    case Isa::kAvx512: return "avx512";
  }
  return "unknown";
}

bool parse_isa_name(const char* text, Isa& out) noexcept {
  if (!text) return false;
  if (std::strcmp(text, "scalar") == 0) {
    out = Isa::kScalar;
    return true;
  }
  if (std::strcmp(text, "avx2") == 0) {
    out = Isa::kAvx2;
    return true;
  }
  if (std::strcmp(text, "avx512") == 0) {
    out = Isa::kAvx512;
    return true;
  }
  return false;
}

namespace {

Isa best_supported() noexcept {
  static const Isa best = detect_isa();
  return best;
}

Isa clamp_isa(Isa isa) noexcept {
  return static_cast<int>(isa) > static_cast<int>(best_supported()) ? best_supported() : isa;
}

Isa env_isa() noexcept {
  static const Isa cached = [] {
    Isa isa;
    const char* e = std::getenv("FPM_FORCE_ISA");
    if (e && parse_isa_name(e, isa)) return clamp_isa(isa);
    return best_supported();
  }();
  return cached;
}

std::atomic<int> g_forced{-1};

struct KernelSet {
  uint64_t (*popcount)(const uint64_t*, size_t) noexcept;
  uint64_t (*intersect_count)(const uint64_t*, const uint64_t*, uint64_t*, size_t) noexcept;
  uint64_t (*intersect_count_nostore)(const uint64_t*, const uint64_t*, size_t) noexcept;
  uint64_t (*intersect_count_nt)(const uint64_t*, const uint64_t*, uint64_t*, size_t) noexcept;
};

#if FPM_X86
const KernelSet kKernelSets[] = {
    {kernels::popcount_scalar, kernels::intersect_count_scalar,
     kernels::intersect_count_nostore_scalar, kernels::intersect_count_scalar},
    {kernels::popcount_avx2, kernels::intersect_count_avx2,
     kernels::intersect_count_nostore_avx2, kernels::intersect_count_nt_avx2},
    {kernels::popcount_avx512, kernels::intersect_count_avx512,
     kernels::intersect_count_nostore_avx512, kernels::intersect_count_nt_avx512},
};
#else
const KernelSet kKernelSets[] = {
    {kernels::popcount_scalar, kernels::intersect_count_scalar,
     kernels::intersect_count_nostore_scalar, kernels::intersect_count_scalar},
    {kernels::popcount_scalar, kernels::intersect_count_scalar,
     kernels::intersect_count_nostore_scalar, kernels::intersect_count_scalar},
    {kernels::popcount_scalar, kernels::intersect_count_scalar,
     kernels::intersect_count_nostore_scalar, kernels::intersect_count_scalar},
};
#endif

const KernelSet& active_kernels() noexcept {
  return kKernelSets[static_cast<int>(current_isa())];
}

}  // namespace

void force_isa(Isa isa) noexcept {
  g_forced.store(static_cast<int>(clamp_isa(isa)), std::memory_order_relaxed);
}

void clear_forced_isa() noexcept { g_forced.store(-1, std::memory_order_relaxed); }

Isa current_isa() noexcept {
  const int f = g_forced.load(std::memory_order_relaxed);
  return f >= 0 ? static_cast<Isa>(f) : env_isa();
}

uint64_t popcount_words(const uint64_t* a, size_t words) noexcept {
  return active_kernels().popcount(a, words);
}

uint64_t intersect_count(const uint64_t* a, const uint64_t* b, uint64_t* out,
                         size_t words) noexcept {
  return active_kernels().intersect_count(a, b, out, words);
}

uint64_t intersect_count_nostore(const uint64_t* a, const uint64_t* b, size_t words) noexcept {
  return active_kernels().intersect_count_nostore(a, b, words);
}

uint64_t intersect_count_nt(const uint64_t* a, const uint64_t* b, uint64_t* out,
                            size_t words) noexcept {
  return active_kernels().intersect_count_nt(a, b, out, words);
}

// --------------------------------------------------------------- allocation ---

uint64_t* alloc_words(size_t words) noexcept {
  if (words == 0) return nullptr;
  const size_t bytes = (words * 8 + 63) & ~size_t(63);
  void* p = std::aligned_alloc(64, bytes);
  if (!p) return nullptr;
  std::memset(p, 0, bytes);
  return static_cast<uint64_t*>(p);
}

void free_words(uint64_t* p) noexcept { std::free(p); }

}  // namespace fpm
