#include "abi.h"

#include <cstring>
#include <exception>
#include <new>
#include <string>

#include "rules.h"

namespace {

thread_local std::string g_last_error;

template <typename T>
T* copy_vec(const std::vector<T>& v) {
  if (v.empty()) return nullptr;
  T* p = new (std::nothrow) T[v.size()];
  if (!p) throw std::bad_alloc();
  std::memcpy(p, v.data(), v.size() * sizeof(T));
  return p;
}

void free_fields(FpmRules* r) {
  delete[] r->ant_offsets;
  delete[] r->ant_keys;
  delete[] r->consequent;
  delete[] r->support;
  delete[] r->confidence;
  delete[] r->lift;
  delete[] r->n_baskets;
  delete[] r->witnesses;
}

}  // namespace

FpmRules* fpm_mine_rules(const uint64_t* offsets, uint64_t offsets_len, const uint64_t* keys,
                         uint64_t keys_len, int64_t min_support_abs, int max_len,
                         int max_antecedents, const double* supp_thr, uint64_t supp_thr_len,
                         const double* conf_thr, uint64_t conf_thr_len, const double* lift_thr,
                         uint64_t lift_thr_len, double min_confidence, int threads,
                         int k_witnesses) {
  try {
    fpm::RuleParams p;
    p.max_antecedents = max_antecedents;
    p.min_confidence = min_confidence;
    p.threads = threads;
    p.k_witnesses = k_witnesses;
    if (supp_thr) p.thr.supp.assign(supp_thr, supp_thr + supp_thr_len);
    if (conf_thr) p.thr.conf.assign(conf_thr, conf_thr + conf_thr_len);
    if (lift_thr) p.thr.lift.assign(lift_thr, lift_thr + lift_thr_len);

    fpm::Rules r = fpm::mine_rules(offsets, static_cast<size_t>(offsets_len), keys,
                                   static_cast<size_t>(keys_len), min_support_abs, max_len, p);

    FpmRules* out = new (std::nothrow) FpmRules{};
    if (!out) throw std::bad_alloc();
    try {
      out->ant_offsets = copy_vec(r.ant_offsets);
      out->ant_keys = copy_vec(r.ant_keys);
      out->consequent = copy_vec(r.consequent);
      out->support = copy_vec(r.support);
      out->confidence = copy_vec(r.confidence);
      out->lift = copy_vec(r.lift);
      out->n_baskets = copy_vec(r.n_baskets);
      out->witnesses = copy_vec(r.witnesses);
    } catch (...) {
      free_fields(out);
      delete out;
      throw;
    }
    out->num_rules = r.size();
    out->num_ant_keys = r.ant_keys.size();
    out->k_witnesses = r.k_witnesses;
    return out;
  } catch (const std::exception& e) {
    g_last_error = e.what();
    return nullptr;
  } catch (...) {
    g_last_error = "unknown error";
    return nullptr;
  }
}

void fpm_free_rules(FpmRules* r) {
  if (!r) return;
  free_fields(r);
  delete r;
}

const char* fpm_last_error(void) { return g_last_error.c_str(); }
