#include "mlx_bindings.hpp"
#include "colmajor_helpers.hpp"
#include <Rcpp.h>

using namespace mlx::core;
using namespace rmlx;

// [[Rcpp::export]]
SEXP cpp_mlx_flatten_r_order(SEXP xp_) {
  array& arr_ref = get_mlx_array(xp_);
  array result = flatten_r_order(arr_ref);
  return make_mlx_xptr(std::move(result));
}
