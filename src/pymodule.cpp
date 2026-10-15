#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "bitset.h"
#include "rules.h"

namespace py = pybind11;

namespace {

using U64Array = py::array_t<uint64_t, py::array::c_style>;
using F64Array = py::array_t<double, py::array::c_style>;

std::vector<double> thr_vec(const F64Array& a, const char* name) {
  if (a.ndim() != 1) throw std::invalid_argument(std::string(name) + " must be 1-D");
  return std::vector<double>(a.data(), a.data() + a.size());
}

template <typename T>
py::array_t<T> to_numpy(const std::vector<T>& v) {
  py::array_t<T> out(v.size());
  if (!v.empty()) std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(T));
  return out;
}

py::dict mine_py(const U64Array& offsets, const U64Array& keys, int64_t min_support_abs,
                 int max_len, int max_antecedents, const F64Array& supp_thr,
                 const F64Array& conf_thr, const F64Array& lift_thr, double min_confidence,
                 int threads, int k_witnesses) {
  if (offsets.ndim() != 1) throw std::invalid_argument("offsets must be 1-D");
  if (keys.ndim() != 1) throw std::invalid_argument("keys must be 1-D");

  fpm::RuleParams p;
  p.max_antecedents = max_antecedents;
  p.min_confidence = min_confidence;
  p.threads = threads;
  p.k_witnesses = k_witnesses;
  p.thr.supp = thr_vec(supp_thr, "supp_thr");
  p.thr.conf = thr_vec(conf_thr, "conf_thr");
  p.thr.lift = thr_vec(lift_thr, "lift_thr");

  fpm::Rules r;
  {
    py::gil_scoped_release release;  // mining runs without the GIL
    r = fpm::mine_rules(offsets.data(), static_cast<size_t>(offsets.size()), keys.data(),
                        static_cast<size_t>(keys.size()), min_support_abs, max_len, p);
  }

  py::dict out;
  out["ant_offsets"] = to_numpy(r.ant_offsets);
  out["ant_keys"] = to_numpy(r.ant_keys);
  out["consequent"] = to_numpy(r.consequent);
  out["support"] = to_numpy(r.support);
  out["confidence"] = to_numpy(r.confidence);
  out["lift"] = to_numpy(r.lift);
  out["n_baskets"] = to_numpy(r.n_baskets);
  py::array_t<int32_t> wit(std::vector<size_t>{r.size(), static_cast<size_t>(r.k_witnesses)});
  if (!r.witnesses.empty())
    std::memcpy(wit.mutable_data(), r.witnesses.data(), r.witnesses.size() * sizeof(int32_t));
  out["witnesses"] = wit;
  return out;
}

py::dict mine_itemsets_py(const U64Array& offsets, const U64Array& keys, int64_t min_support_abs,
                          int max_len, int threads) {
  if (offsets.ndim() != 1) throw std::invalid_argument("offsets must be 1-D");
  if (keys.ndim() != 1) throw std::invalid_argument("keys must be 1-D");

  fpm::FrequentItemsets fi;
  {
    py::gil_scoped_release release;
    fpm::VerticalIndex idx = fpm::VerticalIndex::build(
        offsets.data(), static_cast<size_t>(offsets.size()), keys.data(),
        static_cast<size_t>(keys.size()), min_support_abs, threads);
    fi = fpm::mine_itemsets(idx, max_len, threads);
  }

  py::dict out;
  out["itemset_offsets"] = to_numpy(fi.offsets);
  out["itemset_keys"] = to_numpy(fi.ids);
  out["counts"] = to_numpy(fi.counts);
  out["n_transactions"] = fi.n_transactions;
  return out;
}

}  // namespace

PYBIND11_MODULE(fpm, m) {
  m.doc() = "Multi-core frequent-pattern mining (vertical bitset engine)";
  m.def("mine", &mine_py, py::arg("offsets"), py::arg("keys"), py::arg("min_support_abs"),
        py::arg("max_len"), py::arg("max_antecedents"),
        py::arg("supp_thr") = F64Array(0), py::arg("conf_thr") = F64Array(0),
        py::arg("lift_thr") = F64Array(0), py::arg("min_confidence") = 0.0,
        py::arg("threads") = 0, py::arg("k_witnesses") = 0,
        "Mine association rules from a CSR transaction database.\n\n"
        "offsets: uint64[N+1] CSR row pointers into keys\n"
        "keys: uint64[M] item ids (opaque, caller-defined)\n"
        "min_support_abs: absolute support in transactions\n"
        "max_len: itemset depth cap (e.g. max_antecedents + 1)\n"
        "max_antecedents: rule antecedent-size cap A\n"
        "supp_thr/conf_thr/lift_thr: per-length thresholds, index =\n"
        "    total_size - 1; entries beyond the array length mean no constraint\n"
        "min_confidence: global generation-time floor\n"
        "threads: 0 = auto-detect\n"
        "k_witnesses: witness transactions per rule (0 = skip)");
  m.def("mine_itemsets", &mine_itemsets_py, py::arg("offsets"), py::arg("keys"),
        py::arg("min_support_abs"), py::arg("max_len"), py::arg("threads") = 0,
        "Mine frequent itemsets from a CSR transaction database.\n\n"
        "Returns a dict with itemset_offsets/itemset_keys (CSR) and counts.");
  m.def("detect_isa", [] { return fpm::isa_name(fpm::detect_isa()); },
        "Best ISA the CPU supports (scalar/avx2/avx512).");
  m.def("isa_in_use", [] { return fpm::isa_name(fpm::current_isa()); },
        "ISA actually dispatching (honors FPM_FORCE_ISA).");
}
