#include "mlx_bindings.hpp"
#include "colmajor_helpers.hpp"
#include <Rcpp.h>

using namespace mlx::core;
using namespace rmlx;

// [[Rcpp::export]]
SEXP cpp_mlx_flatten_r_order(SEXP xp_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array result = flatten_r_order(wrapper->get());
  return make_mlx_xptr(std::move(result));
}
