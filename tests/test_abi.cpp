#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "abi.h"
#include "rules.h"

namespace {

TEST(Abi, MatchesCppApi) {
  const std::vector<uint64_t> offsets = {0, 3, 5, 7, 9, 13};
  const std::vector<uint64_t> keys = {1, 2, 3, 1, 2, 1, 3, 2, 3, 1, 2, 3, 4};

  FpmRules* r = fpm_mine_rules(offsets.data(), offsets.size(), keys.data(), keys.size(),
                               /*min_support_abs=*/2, /*max_len=*/3, /*max_antecedents=*/2,
                               /*supp_thr=*/nullptr, 0, /*conf_thr=*/nullptr, 0,
                               /*lift_thr=*/nullptr, 0, /*min_confidence=*/0.0,
                               /*threads=*/1, /*k_witnesses=*/3);
  ASSERT_NE(r, nullptr) << fpm_last_error();
  EXPECT_GT(r->num_rules, 0u);
  EXPECT_EQ(r->k_witnesses, 3);
  EXPECT_EQ(r->ant_offsets[0], 0u);
  EXPECT_EQ(r->ant_offsets[r->num_rules], r->num_ant_keys);
  for (uint64_t i = 0; i < r->num_rules; ++i) {
    EXPECT_GT(r->confidence[i], 0.0);
    EXPECT_LE(r->confidence[i], 1.0);
    EXPECT_GE(r->n_baskets[i], 2);
    EXPECT_LE(r->n_baskets[i], 3);
  }
  fpm_free_rules(r);
}

TEST(Abi, RepeatedCallsNoLeak) {
  const std::vector<uint64_t> offsets = {0, 2, 4, 6};
  const std::vector<uint64_t> keys = {1, 2, 2, 3, 1, 2};
  for (int iter = 0; iter < 100; ++iter) {
    FpmRules* r = fpm_mine_rules(offsets.data(), offsets.size(), keys.data(), keys.size(), 1, 2,
                                 1, nullptr, 0, nullptr, 0, nullptr, 0, 0.0, 1, 2);
    ASSERT_NE(r, nullptr);
    fpm_free_rules(r);
  }
  fpm_free_rules(nullptr);  // must be safe
}

TEST(Abi, ErrorPropagation) {
  const std::vector<uint64_t> offsets = {0, 2, 4, 6};
  const std::vector<uint64_t> keys = {1, 2, 2, 3, 1, 2};
  // min_support_abs = 0 -> invalid
  FpmRules* r = fpm_mine_rules(offsets.data(), offsets.size(), keys.data(), keys.size(), 0, 2, 1,
                               nullptr, 0, nullptr, 0, nullptr, 0, 0.0, 1, 0);
  EXPECT_EQ(r, nullptr);
  EXPECT_STRNE(fpm_last_error(), "");
  // malformed CSR
  r = fpm_mine_rules(offsets.data(), offsets.size(), keys.data(), 999, 1, 2, 1, nullptr, 0,
                     nullptr, 0, nullptr, 0, 0.0, 1, 0);
  EXPECT_EQ(r, nullptr);
  EXPECT_STRNE(fpm_last_error(), "");
}

TEST(Abi, ThresholdArraysPassedThrough) {
  const std::vector<uint64_t> offsets = {0, 3, 5, 7, 9, 13};
  const std::vector<uint64_t> keys = {1, 2, 3, 1, 2, 1, 3, 2, 3, 1, 2, 3, 4};
  const std::vector<double> conf_thr = {0.0, 0.999};  // size-2 rules need conf >= 0.999
  FpmRules* r = fpm_mine_rules(offsets.data(), offsets.size(), keys.data(), keys.size(), 2, 3, 2,
                               nullptr, 0, conf_thr.data(), conf_thr.size(), nullptr, 0, 0.0, 1,
                               0);
  ASSERT_NE(r, nullptr) << fpm_last_error();
  for (uint64_t i = 0; i < r->num_rules; ++i) {
    const uint64_t ant_len = r->ant_offsets[i + 1] - r->ant_offsets[i];
    if (ant_len == 1) {
      EXPECT_GE(r->confidence[i], 0.999);
    }
  }
  fpm_free_rules(r);
}

}  // namespace
