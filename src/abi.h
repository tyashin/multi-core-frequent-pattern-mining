#ifndef FPM_ABI_H
#define FPM_ABI_H

/* Minimal extern "C" entry point for non-Python consumers / ctypes.
 * The core library knows nothing about this wrapper; errors are reported
 * as nullptr + fpm_last_error() instead of exceptions.
 *
 * Usage:
 *   FpmRules* r = fpm_mine_rules(...);
 *   if (!r) { fprintf(stderr, "%s\n", fpm_last_error()); }
 *   ... use r->ant_offsets[r->num_rules] etc. ...
 *   fpm_free_rules(r);
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FpmRules {
  uint64_t* ant_offsets;  /* num_rules + 1 */
  uint64_t* ant_keys;     /* num_ant_keys */
  uint64_t* consequent;   /* num_rules */
  double* support;        /* num_rules */
  double* confidence;     /* num_rules */
  double* lift;           /* num_rules */
  int64_t* n_baskets;     /* num_rules */
  int32_t* witnesses;     /* num_rules * k_witnesses, -1 padded; NULL when k == 0 */
  uint64_t num_rules;
  uint64_t num_ant_keys;
  int k_witnesses;
} FpmRules;

/* Threshold arrays are indexed by size_key = min(total_size, 5) - 1;
 * pass a shorter array (or NULL with length 0) for missing keys = 0.0.
 * threads: 0 = auto-detect. Returns NULL on error (see fpm_last_error). */
FpmRules* fpm_mine_rules(const uint64_t* offsets, uint64_t offsets_len, const uint64_t* keys,
                         uint64_t keys_len, int64_t min_support_abs, int max_len,
                         int max_antecedents, const double* supp_thr, uint64_t supp_thr_len,
                         const double* conf_thr, uint64_t conf_thr_len, const double* lift_thr,
                         uint64_t lift_thr_len, double min_confidence, int threads,
                         int k_witnesses);

void fpm_free_rules(FpmRules* r);

/* Thread-local error message, valid until the next fpm call on this thread. */
const char* fpm_last_error(void);

#ifdef __cplusplus
}
#endif

#endif  /* FPM_ABI_H */
