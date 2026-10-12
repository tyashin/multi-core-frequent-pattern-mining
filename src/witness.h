#pragma once

#include <cstdint>

#include "mine.h"
#include "rules.h"

// Bounded witness extraction: for each surviving rule, up to k transaction
// indices whose transaction contains every rule item (antecedent ∪
// consequent). Union bitsets are never retained; each rule re-intersects its
// (<= max_len) item rows word-by-word and stops after k set bits.

namespace fpm {

// Fills rules.witnesses (size R * k, -1 padded) in place.
void fill_witnesses(const VerticalIndex& db, Rules& rules, int k, int threads);

}  // namespace fpm
