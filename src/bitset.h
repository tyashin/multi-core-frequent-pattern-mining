#pragma once

#include <cstddef>
#include <cstdint>

// Vertical bitset kernels: intersect + popcount over uint64_t word arrays.
// Three ISA variants (scalar / AVX2 / AVX-512) behind runtime dispatch.
//
// Buffer contract: rows passed to the *_nt (non-temporal store) kernels must
// be 64-byte aligned; alloc_words() provides that. All kernels also handle
// unaligned input rows and a tail of words % vector_width.

namespace fpm {

enum class Isa { kScalar = 0, kAvx2 = 1, kAvx512 = 2 };

// Best ISA the running CPU supports.
Isa detect_isa() noexcept;

// Effective ISA: test hook (force_isa) > FPM_FORCE_ISA env var > detect_isa().
// A request for an ISA the CPU lacks is clamped down to the best supported.
Isa current_isa() noexcept;

// Test/CLI hook. Clamped against CPU support; clear with clear_forced_isa().
void force_isa(Isa isa) noexcept;
void clear_forced_isa() noexcept;

// Parses "scalar" | "avx2" | "avx512"; returns false on unknown text.
bool parse_isa_name(const char* text, Isa& out) noexcept;

const char* isa_name(Isa isa) noexcept;

// Rounds a word count up to a 64-byte multiple (8 words) so that rows of
// this stride keep every row 64-byte aligned when the base is aligned.
inline size_t padded_words(size_t words) noexcept { return (words + 7) & ~size_t(7); }

// 64-byte aligned, zero-initialized buffer of `words` uint64_t.
// Returns nullptr on allocation failure.
uint64_t* alloc_words(size_t words) noexcept;
void free_words(uint64_t* p) noexcept;

namespace kernels {

uint64_t popcount_scalar(const uint64_t* a, size_t words) noexcept;
uint64_t popcount_avx2(const uint64_t* a, size_t words) noexcept;
uint64_t popcount_avx512(const uint64_t* a, size_t words) noexcept;

// out[i] = a[i] & b[i]; returns popcount of the intersection.
uint64_t intersect_count_scalar(const uint64_t* a, const uint64_t* b, uint64_t* out, size_t words) noexcept;
uint64_t intersect_count_avx2(const uint64_t* a, const uint64_t* b, uint64_t* out, size_t words) noexcept;
uint64_t intersect_count_avx512(const uint64_t* a, const uint64_t* b, uint64_t* out, size_t words) noexcept;

// Count only, no store (terminal-level store-skip).
uint64_t intersect_count_nostore_scalar(const uint64_t* a, const uint64_t* b, size_t words) noexcept;
uint64_t intersect_count_nostore_avx2(const uint64_t* a, const uint64_t* b, size_t words) noexcept;
uint64_t intersect_count_nostore_avx512(const uint64_t* a, const uint64_t* b, size_t words) noexcept;

// Non-temporal-store variants (write-once, read-later child bitsets).
// `out` must be 64-byte aligned. Ends with a store fence.
uint64_t intersect_count_nt_avx2(const uint64_t* a, const uint64_t* b, uint64_t* out, size_t words) noexcept;
uint64_t intersect_count_nt_avx512(const uint64_t* a, const uint64_t* b, uint64_t* out, size_t words) noexcept;

}  // namespace kernels

// Dispatched entry points (route via current_isa()).
uint64_t popcount_words(const uint64_t* a, size_t words) noexcept;
uint64_t intersect_count(const uint64_t* a, const uint64_t* b, uint64_t* out, size_t words) noexcept;
uint64_t intersect_count_nostore(const uint64_t* a, const uint64_t* b, size_t words) noexcept;
// Falls back to the regular store kernel when the CPU has no vector ISA
// (scalar has no NT store) — result is identical either way.
uint64_t intersect_count_nt(const uint64_t* a, const uint64_t* b, uint64_t* out, size_t words) noexcept;

}  // namespace fpm
