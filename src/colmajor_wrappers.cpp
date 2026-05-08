#include "mlx_bindings.hpp"
#include "colmajor_helpers.hpp"
#include <Rcpp.h>

using namespace mlx::core;
using namespace rmlx;

// [[Rcpp::export]]
SEXP cpp_mlx_flatten_r_order(SEXP xp_, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();
  StreamOrDevice dev = typed_device(arr.dtype(), device_str);
  array result = flatten_r_order(arr, dev);
  return make_mlx_xptr(std::move(result));
}
