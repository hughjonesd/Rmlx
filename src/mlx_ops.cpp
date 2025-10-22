#include "mlx_bindings.hpp"
#include <mlx/mlx.h>
#include <mlx/linalg.h>
#include <mlx/fft.h>
#include <mlx/random.h>
#include <Rcpp.h>
#include <string>
#include <numeric>
#include <algorithm>
#include <limits>
#include <set>
#include <cstdint>
#include <complex>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

namespace {

int normalize_axis(const array& arr, int axis) {
  int ndim = static_cast<int>(arr.ndim());
  if (axis < 0) {
    axis += ndim;
  }
  if (axis < 0 || axis >= ndim) {
    Rcpp::stop("Axis %d is out of bounds for array with %d dimensions.", axis + 1, ndim);
  }
  return axis;
}

std::vector<int> normalize_axes(const array& arr, const std::vector<int>& axes) {
  std::vector<int> result;
  result.reserve(axes.size());
  for (int axis : axes) {
    result.push_back(normalize_axis(arr, axis));
  }
  return result;
}

std::optional<std::vector<int>> optional_axes(
    const array& arr,
    const Rcpp::Nullable<Rcpp::IntegerVector>& axes) {
  if (axes.isNull()) {
    return std::nullopt;
  }
  Rcpp::IntegerVector axes_vec(axes.get());
  if (axes_vec.size() == 0) {
    Rcpp::stop("axes must contain at least one element.");
  }
  std::vector<int> raw_axes(axes_vec.begin(), axes_vec.end());
  for (int& axis : raw_axes) {
    if (axis == 0) {
      Rcpp::stop("Axis indices are 1-based; axis 0 is invalid.");
    }
    if (axis > 0) {
      axis -= 1;
    }
  }
  return normalize_axes(arr, raw_axes);
}

Dtype promote_numeric_dtype(Dtype lhs, Dtype rhs) {
  if (lhs == complex64 || rhs == complex64) {
    return complex64;
  }
  if (lhs == float64 || rhs == float64) {
    return float64;
  }
  if (lhs == float32 || rhs == float32) {
    return float32;
  }
  if (lhs == bool_ && rhs == bool_) {
    return bool_;
  }
  if ((lhs == bool_ && rhs == int32) || (rhs == bool_ && lhs == int32)) {
    return int32;
  }
  if ((lhs == bool_ && rhs == int64) || (rhs == bool_ && lhs == int64)) {
    return int64;
  }
  return float32;
}

std::vector<int> normalize_new_axes(const array& arr, const std::vector<int>& axes) {
  int ndim = static_cast<int>(arr.ndim());
  std::vector<int> normalized;
  normalized.reserve(axes.size());
  std::set<int> seen;
  for (int axis : axes) {
    int ax = axis;
    if (ax < 0) {
      ax += ndim + 1;
    }
    if (ax < 0 || ax > ndim) {
      Rcpp::stop("axis %d is out of bounds for array with %d dimensions.", axis, ndim);
    }
    if (!seen.insert(ax).second) {
      Rcpp::stop("axis %d is repeated.", axis);
    }
    normalized.push_back(ax);
  }
  std::sort(normalized.begin(), normalized.end());
  return normalized;
}

}  // namespace

// Unary operations
// [[Rcpp::export]]
SEXP cpp_mlx_unary(SEXP xp_, std::string op) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  array result = [&]() -> array {
    if (op == "neg") {
      return negative(wrapper->get());
    } else if (op == "abs") {
      return abs(wrapper->get());
    } else if (op == "sign") {
      return sign(wrapper->get());
    } else if (op == "sqrt") {
      return sqrt(wrapper->get());
    } else if (op == "rsqrt") {
      return rsqrt(wrapper->get());
    } else if (op == "square") {
      return square(wrapper->get());
    } else if (op == "exp") {
      return exp(wrapper->get());
    } else if (op == "expm1") {
      return expm1(wrapper->get());
    } else if (op == "log") {
      return log(wrapper->get());
    } else if (op == "log2") {
      return log2(wrapper->get());
    } else if (op == "log10") {
      return log10(wrapper->get());
    } else if (op == "log1p") {
      return log1p(wrapper->get());
    } else if (op == "sin") {
      return sin(wrapper->get());
    } else if (op == "cos") {
      return cos(wrapper->get());
    } else if (op == "tan") {
      return tan(wrapper->get());
    } else if (op == "asin") {
      return arcsin(wrapper->get());
    } else if (op == "acos") {
      return arccos(wrapper->get());
    } else if (op == "atan") {
      return arctan(wrapper->get());
    } else if (op == "sinh") {
      return sinh(wrapper->get());
    } else if (op == "cosh") {
      return cosh(wrapper->get());
    } else if (op == "tanh") {
      return tanh(wrapper->get());
    } else if (op == "asinh") {
      return arcsinh(wrapper->get());
    } else if (op == "acosh") {
      return arccosh(wrapper->get());
    } else if (op == "atanh") {
      return arctanh(wrapper->get());
    } else if (op == "floor") {
      return floor(wrapper->get());
    } else if (op == "ceil") {
      return ceil(wrapper->get());
    } else if (op == "round") {
      return round(wrapper->get());
    } else {
      Rcpp::stop("Unsupported unary operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_logical_not(SEXP xp_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();
  array arr_bool = astype(arr, bool_);
  array result = logical_not(arr_bool);
  return make_mlx_xptr(std::move(result));
}

// Binary operations
// [[Rcpp::export]]
SEXP cpp_mlx_binary(SEXP xp1_, SEXP xp2_, std::string op,
                    std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  lhs = astype(lhs, target_dtype);
  rhs = astype(rhs, target_dtype);

  lhs = astype(lhs, target_dtype, target_device);
  rhs = astype(rhs, target_dtype, target_device);

  array result = [&]() -> array {
    if (op == "+") {
      return add(lhs, rhs);
    } else if (op == "-") {
      return subtract(lhs, rhs);
    } else if (op == "*") {
      return multiply(lhs, rhs);
    } else if (op == "/") {
      return divide(lhs, rhs);
    } else if (op == "^") {
      return power(lhs, rhs);
    } else if (op == "==") {
      return equal(lhs, rhs);
    } else if (op == "!=") {
      return not_equal(lhs, rhs);
    } else if (op == "<") {
      return less(lhs, rhs);
    } else if (op == "<=") {
      return less_equal(lhs, rhs);
    } else if (op == ">") {
      return greater(lhs, rhs);
    } else if (op == ">=") {
      return greater_equal(lhs, rhs);
    } else {
      Rcpp::stop("Unsupported binary operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_minimum(SEXP xp1_, SEXP xp2_, std::string device_str) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  StreamOrDevice target_device = string_to_device(device_str);

  Dtype target_dtype = lhs.dtype();
  if (target_dtype == bool_) {
    target_dtype = float32;
  }
  if (rhs.dtype() == float64 || target_dtype == float64) {
    target_dtype = float64;
  } else if (rhs.dtype() == float32 || target_dtype == float32) {
    target_dtype = float32;
  }

  lhs = astype(lhs, target_dtype, target_device);
  rhs = astype(rhs, target_dtype, target_device);

  array result = minimum(lhs, rhs, target_device);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_maximum(SEXP xp1_, SEXP xp2_, std::string device_str) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  StreamOrDevice target_device = string_to_device(device_str);

  Dtype target_dtype = lhs.dtype();
  if (target_dtype == bool_) {
    target_dtype = float32;
  }
  if (rhs.dtype() == float64 || target_dtype == float64) {
    target_dtype = float64;
  } else if (rhs.dtype() == float32 || target_dtype == float32) {
    target_dtype = float32;
  }

  lhs = astype(lhs, target_dtype, target_device);
  rhs = astype(rhs, target_dtype, target_device);

  array result = maximum(lhs, rhs, target_device);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_clip(SEXP xp_, SEXP min_, SEXP max_, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice target_device = string_to_device(device_str);
  Dtype original_dtype = arr.dtype();

  if (!(original_dtype == float32 || original_dtype == float64)) {
    original_dtype = float32;
    arr = astype(arr, original_dtype, target_device);
  } else {
    arr = astype(arr, original_dtype, target_device);
  }

  double min_val = Rf_isNull(min_) ? -std::numeric_limits<double>::infinity()
                                   : Rcpp::as<double>(min_);
  double max_val = Rf_isNull(max_) ? std::numeric_limits<double>::infinity()
                                   : Rcpp::as<double>(max_);

  if (min_val > max_val) {
    Rcpp::stop("min must be less than or equal to max.");
  }

  array min_arr = array(min_val, original_dtype);
  array max_arr = array(max_val, original_dtype);

  min_arr = astype(min_arr, original_dtype, target_device);
  max_arr = astype(max_arr, original_dtype, target_device);

  array result = clip(arr, min_arr, max_arr, target_device);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_floor_divide(SEXP xp1_, SEXP xp2_, std::string device_str) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  StreamOrDevice target_device = string_to_device(device_str);
  Dtype target_dtype = promote_numeric_dtype(lhs.dtype(), rhs.dtype());

  lhs = astype(lhs, target_dtype, target_device);
  rhs = astype(rhs, target_dtype, target_device);

  array result = floor_divide(lhs, rhs, target_device);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_remainder(SEXP xp1_, SEXP xp2_, std::string device_str) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  StreamOrDevice target_device = string_to_device(device_str);
  Dtype target_dtype = promote_numeric_dtype(lhs.dtype(), rhs.dtype());

  lhs = astype(lhs, target_dtype, target_device);
  rhs = astype(rhs, target_dtype, target_device);

  array result = remainder(lhs, rhs, target_device);
  return make_mlx_xptr(std::move(result));
}

// Logical operations
// [[Rcpp::export]]
SEXP cpp_mlx_logical(SEXP xp1_, SEXP xp2_, std::string op, std::string device_str) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  StreamOrDevice target_device = string_to_device(device_str);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  lhs = astype(lhs, bool_, target_device);
  rhs = astype(rhs, bool_, target_device);

  array result = [&]() -> array {
    if (op == "&" || op == "&&") {
      return logical_and(lhs, rhs, target_device);
    } else if (op == "|" || op == "||") {
      return logical_or(lhs, rhs, target_device);
    } else {
      Rcpp::stop("Unsupported logical operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// Reductions
// [[Rcpp::export]]
SEXP cpp_mlx_reduce(SEXP xp_, std::string op, int ddof) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  array input = wrapper->get();
  array result = [&]() -> array {
    if (op == "sum") {
      if (input.dtype() == bool_) {
        input = astype(input, float32);
      }
      return sum(input, false);
    } else if (op == "prod") {
      if (input.dtype() == bool_) {
        input = astype(input, float32);
      }
      return prod(input, false);
    } else if (op == "mean") {
      if (input.dtype() == bool_) {
        input = astype(input, float32);
      }
      return mean(input, false);
    } else if (op == "var") {
      if (input.dtype() == bool_ || input.dtype() == int32 || input.dtype() == int64) {
        input = astype(input, float32);
      }
      Dtype out_dtype = (input.dtype() == float64) ? float64 : float32;
      input = astype(input, out_dtype);
      return mlx::core::var(input, false, ddof);
    } else if (op == "std") {
      if (input.dtype() == bool_ || input.dtype() == int32 || input.dtype() == int64) {
        input = astype(input, float32);
      }
      Dtype out_dtype = (input.dtype() == float64) ? float64 : float32;
      input = astype(input, out_dtype);
      return mlx::core::std(input, false, ddof);
    } else if (op == "min") {
      return min(input, false);
    } else if (op == "max") {
      return max(input, false);
    } else if (op == "all") {
      array bool_input = astype(input, bool_);
      return all(bool_input, false);
    } else if (op == "any") {
      array bool_input = astype(input, bool_);
      return any(bool_input, false);
    } else {
      Rcpp::stop("Unsupported reduction operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// Axis reductions
// [[Rcpp::export]]
SEXP cpp_mlx_reduce_axis(SEXP xp_, std::string op, int axis, bool keepdims, int ddof) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  std::vector<int> axes = {axis};
  array input = wrapper->get();

  array result = [&]() -> array {
    if (op == "sum") {
      if (input.dtype() == bool_) {
        input = astype(input, float32);
      }
      return sum(input, axes, keepdims);
    } else if (op == "prod") {
      if (input.dtype() == bool_) {
        input = astype(input, float32);
      }
      return prod(input, axes, keepdims);
    } else if (op == "mean") {
      if (input.dtype() == bool_) {
        input = astype(input, float32);
      }
      return mean(input, axes, keepdims);
    } else if (op == "var") {
      if (input.dtype() == bool_ || input.dtype() == int32 || input.dtype() == int64) {
        input = astype(input, float32);
      }
      Dtype out_dtype = (input.dtype() == float64) ? float64 : float32;
      input = astype(input, out_dtype);
      return mlx::core::var(input, axes, keepdims, ddof);
    } else if (op == "std") {
      if (input.dtype() == bool_ || input.dtype() == int32 || input.dtype() == int64) {
        input = astype(input, float32);
      }
      Dtype out_dtype = (input.dtype() == float64) ? float64 : float32;
      input = astype(input, out_dtype);
      return mlx::core::std(input, axes, keepdims, ddof);
    } else if (op == "min") {
      return min(input, axes, keepdims);
    } else if (op == "max") {
      return max(input, axes, keepdims);
    } else if (op == "all") {
      array bool_input = astype(input, bool_);
      return all(bool_input, axes, keepdims);
    } else if (op == "any") {
      array bool_input = astype(input, bool_);
      return any(bool_input, axes, keepdims);
    } else {
      Rcpp::stop("Unsupported axis reduction operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// Arg reductions
// [[Rcpp::export]]
SEXP cpp_mlx_argmax(SEXP xp_, Rcpp::Nullable<int> axis, bool keepdims) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int ax = Rcpp::as<int>(axis.get());
      ax = normalize_axis(arr, ax);
      return argmax(arr, ax, keepdims);
    }
    return argmax(arr, keepdims);
  }();

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_argmin(SEXP xp_, Rcpp::Nullable<int> axis, bool keepdims) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int ax = Rcpp::as<int>(axis.get());
      ax = normalize_axis(arr, ax);
      return argmin(arr, ax, keepdims);
    }
    return argmin(arr, keepdims);
  }();

  return make_mlx_xptr(std::move(result));
}

// Sorting helpers
// [[Rcpp::export]]
SEXP cpp_mlx_sort(SEXP xp_, Rcpp::Nullable<int> axis) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int ax = Rcpp::as<int>(axis.get());
      ax = normalize_axis(arr, ax);
      return sort(arr, ax);
    }
    return sort(arr);
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_argsort(SEXP xp_, Rcpp::Nullable<int> axis) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int ax = Rcpp::as<int>(axis.get());
      ax = normalize_axis(arr, ax);
      return argsort(arr, ax);
    }
    return argsort(arr);
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_topk(SEXP xp_, int k, Rcpp::Nullable<int> axis) {
  if (k <= 0) {
    Rcpp::stop("k must be positive.");
  }

  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int ax = Rcpp::as<int>(axis.get());
      ax = normalize_axis(arr, ax);
      int axis_size = static_cast<int>(arr.shape()[ax]);
      if (k > axis_size) {
        Rcpp::stop("k (%d) exceeds size of axis %d (%d).", k, ax + 1, axis_size);
      }
      return topk(arr, k, ax);
    } else {
      int total = static_cast<int>(arr.size());
      if (k > total) {
        Rcpp::stop("k (%d) exceeds number of elements (%d).", k, total);
      }
      return topk(arr, k);
    }
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_partition(SEXP xp_, int kth, Rcpp::Nullable<int> axis) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  if (kth < 0) {
    Rcpp::stop("kth must be non-negative.");
  }

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int target_axis = normalize_axis(arr, Rcpp::as<int>(axis.get()));
      int axis_size = static_cast<int>(arr.shape()[target_axis]);
      if (kth >= axis_size) {
        Rcpp::stop("kth (%d) exceeds size of axis %d (%d).", kth, target_axis + 1, axis_size);
      }
      return partition(arr, kth, target_axis);
    }
    int total = static_cast<int>(arr.size());
    if (kth >= total) {
      Rcpp::stop("kth (%d) exceeds number of elements (%d).", kth, total);
    }
    return partition(arr, kth);
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_argpartition(SEXP xp_, int kth, Rcpp::Nullable<int> axis) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  if (kth < 0) {
    Rcpp::stop("kth must be non-negative.");
  }

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int target_axis = normalize_axis(arr, Rcpp::as<int>(axis.get()));
      int axis_size = static_cast<int>(arr.shape()[target_axis]);
      if (kth >= axis_size) {
        Rcpp::stop("kth (%d) exceeds size of axis %d (%d).", kth, target_axis + 1, axis_size);
      }
      return argpartition(arr, kth, target_axis);
    }
    int total = static_cast<int>(arr.size());
    if (kth >= total) {
      Rcpp::stop("kth (%d) exceeds number of elements (%d).", kth, total);
    }
    return argpartition(arr, kth);
  }();
  return make_mlx_xptr(std::move(result));
}

// logsumexp / logcumsumexp / softmax helpers
// [[Rcpp::export]]
SEXP cpp_mlx_logsumexp(SEXP xp_, Rcpp::Nullable<Rcpp::IntegerVector> axes,
                       bool keepdims) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = [&]() -> array {
    if (axes.isNotNull()) {
      Rcpp::IntegerVector axes_vec(axes.get());
      std::vector<int> ax(axes_vec.begin(), axes_vec.end());
      return logsumexp(arr, normalize_axes(arr, ax), keepdims);
    }
    return logsumexp(arr, keepdims);
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_logsumexp_axis(SEXP xp_, int axis, bool keepdims) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();
  int ax = normalize_axis(arr, axis);
  array result = logsumexp(arr, ax, keepdims);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_logcumsumexp(SEXP xp_, Rcpp::Nullable<int> axis,
                          bool reverse, bool inclusive) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int ax = normalize_axis(arr, Rcpp::as<int>(axis.get()));
      return logcumsumexp(arr, ax, reverse, inclusive);
    }
    return logcumsumexp(arr, reverse, inclusive);
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_softmax(SEXP xp_, Rcpp::Nullable<Rcpp::IntegerVector> axes,
                     bool precise) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = [&]() -> array {
    if (axes.isNotNull()) {
      Rcpp::IntegerVector axes_vec(axes.get());
      std::vector<int> ax(axes_vec.begin(), axes_vec.end());
      return softmax(arr, normalize_axes(arr, ax), precise);
    }
    return softmax(arr, precise);
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_softmax_axis(SEXP xp_, int axis, bool precise) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();
  int ax = normalize_axis(arr, axis);
  array result = softmax(arr, ax, precise);
  return make_mlx_xptr(std::move(result));
}

// Matrix operations
// [[Rcpp::export]]
SEXP cpp_mlx_transpose(SEXP xp_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array result = transpose(wrapper->get());
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_reshape(SEXP xp_, SEXP new_dim_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  IntegerVector new_dim(new_dim_);

  Shape shape(new_dim.begin(), new_dim.end());
  array result = reshape(wrapper->get(), shape);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_matmul(SEXP xp1_, SEXP xp2_,
                    std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  lhs = astype(lhs, target_dtype);
  rhs = astype(rhs, target_dtype);

  lhs = astype(lhs, target_dtype, target_device);
  rhs = astype(rhs, target_dtype, target_device);

  array result = matmul(lhs, rhs);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_cast(SEXP xp_, std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  Dtype dtype = string_to_dtype(dtype_str);
  StreamOrDevice dev = string_to_device(device_str);

  array arr = wrapper->get();
  array result = astype(arr, dtype, dev);
  return make_mlx_xptr(std::move(result));
}

// Array creation helpers
// [[Rcpp::export]]
SEXP cpp_mlx_zeros(SEXP dim_, std::string dtype_str, std::string device_str) {
  IntegerVector dim(dim_);
  if (dim.size() == 0) {
    Rcpp::stop("dim must contain at least one element.");
  }

  Shape shape(dim.begin(), dim.end());
  Dtype dtype = string_to_dtype(dtype_str);
  StreamOrDevice dev = string_to_device(device_str);

  array result = zeros(shape, dtype, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_ones(SEXP dim_, std::string dtype_str, std::string device_str) {
  IntegerVector dim(dim_);
  if (dim.size() == 0) {
    Rcpp::stop("dim must contain at least one element.");
  }

  Shape shape(dim.begin(), dim.end());
  Dtype dtype = string_to_dtype(dtype_str);
  StreamOrDevice dev = string_to_device(device_str);

  array result = ones(shape, dtype, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_full(SEXP dim_, SEXP value_, std::string dtype_str, std::string device_str) {
  if (Rf_length(value_) != 1) {
    Rcpp::stop("value must be a scalar.");
  }

  IntegerVector dim(dim_);
  if (dim.size() == 0) {
    Rcpp::stop("dim must contain at least one element.");
  }

  Shape shape(dim.begin(), dim.end());
  Dtype dtype = string_to_dtype(dtype_str);
  StreamOrDevice dev = string_to_device(device_str);

  array result = [&]() -> array {
    if (dtype == complex64) {
      Rcomplex val = Rcpp::as<Rcomplex>(value_);
      std::complex<float> cval(static_cast<float>(val.r), static_cast<float>(val.i));
      return full(shape, cval, complex64, dev);
    } else if (dtype == bool_) {
      bool val = Rcpp::as<bool>(value_);
      return full(shape, val, bool_, dev);
    } else if (dtype == float64) {
      double val = Rcpp::as<double>(value_);
      return full(shape, val, float64, dev);
    } else if (dtype == float32) {
      double val = Rcpp::as<double>(value_);
      return full(shape, static_cast<float>(val), float32, dev);
    } else {
      Rcpp::stop("Unsupported dtype for mlx_full.");
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_eye(int n,
                 Rcpp::Nullable<int> m,
                 int k,
                 std::string dtype_str,
                 std::string device_str) {
  if (n <= 0) {
    Rcpp::stop("n must be positive.");
  }
  int m_val = m.isNull() ? n : Rcpp::as<int>(m.get());
  if (m_val <= 0) {
    Rcpp::stop("m must be positive.");
  }

  Dtype dtype = string_to_dtype(dtype_str);
  StreamOrDevice dev = string_to_device(device_str);

  array result = eye(n, m_val, k, dtype, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_identity(int n, std::string dtype_str, std::string device_str) {
  if (n <= 0) {
    Rcpp::stop("n must be positive.");
  }
  Dtype dtype = string_to_dtype(dtype_str);
  StreamOrDevice dev = string_to_device(device_str);

  array result = identity(n, dtype, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_arange(SEXP start_,
                    double stop,
                    SEXP step_,
                    std::string dtype_str,
                    std::string device_str) {
  Dtype dtype = string_to_dtype(dtype_str);
  if (!(dtype == float32 || dtype == float64)) {
    Rcpp::stop("mlx_arange currently supports float32 or float64 dtypes.");
  }
  StreamOrDevice dev = string_to_device(device_str);

  bool has_start = start_ != R_NilValue;
  bool has_step = step_ != R_NilValue;

  array result = [&]() -> array {
    double start_val = has_start ? Rcpp::as<double>(start_) : 0.0;
    double step_val = has_step ? Rcpp::as<double>(step_) : 1.0;

    if (has_start && has_step) {
      return arange(start_val, stop, step_val, dtype, dev);
    } else if (has_start) {
      return arange(start_val, stop, dtype, dev);
    } else if (has_step) {
      return arange(0.0, stop, step_val, dtype, dev);
    }
    return arange(stop, dtype, dev);
  }();

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_linspace(double start,
                      double stop,
                      int num,
                      std::string dtype_str,
                      std::string device_str) {
  if (num <= 0) {
    Rcpp::stop("num must be positive.");
  }
  Dtype dtype = string_to_dtype(dtype_str);
  StreamOrDevice dev = string_to_device(device_str);

  array result = linspace(start, stop, num, dtype, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_normal(SEXP dim_, double mean, double std,
                           std::string dtype_str, std::string device_str) {
  IntegerVector dim(dim_);
  Shape shape(dim.begin(), dim.end());

  Dtype dtype = string_to_dtype(dtype_str);
  if (dtype != float32 && dtype != float64) {
    Rcpp::stop("Random normal currently supports dtype = \"float32\" or \"float64\" only.");
  }
  StreamOrDevice dev_input = string_to_device(device_str);
  array result = mlx::core::random::normal(shape, dtype, array(static_cast<float>(mean), dtype), array(static_cast<float>(std), dtype), std::nullopt, dev_input);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_bernoulli(SEXP dim_, double prob, std::string device_str) {
  if (prob < 0.0 || prob > 1.0) {
    Rcpp::stop("prob must be between 0 and 1.");
  }
  IntegerVector dim(dim_);
  Shape shape(dim.begin(), dim.end());
  StreamOrDevice dev_input = string_to_device(device_str);
  array result = mlx::core::random::bernoulli(array(static_cast<float>(prob), float32), shape, std::nullopt, dev_input);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_concat(SEXP args_, int axis) {
  List args(args_);
  if (args.size() == 0) {
    Rcpp::stop("No tensors supplied for concatenation.");
  }
  std::vector<array> arrays;
  arrays.reserve(args.size());
  std::string device_str;
  for (int i = 0; i < args.size(); ++i) {
    List obj(args[i]);
    arrays.push_back(get_mlx_wrapper(obj["ptr"])->get());
    if (i == 0) {
      device_str = Rcpp::as<std::string>(obj["device"]);
    }
  }

  StreamOrDevice dev = string_to_device(device_str);
  array result = concatenate(arrays, axis, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_stack(SEXP args_, int axis, std::string device_str) {
  List args(args_);
  if (args.size() == 0) {
    Rcpp::stop("No tensors supplied for stacking.");
  }

  std::vector<array> arrays;
  arrays.reserve(args.size());
  for (int i = 0; i < args.size(); ++i) {
    List obj(args[i]);
    arrays.push_back(get_mlx_wrapper(obj["ptr"])->get());
  }

  StreamOrDevice dev = string_to_device(device_str);
  array result = stack(arrays, axis, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_squeeze(SEXP xp_, Rcpp::Nullable<Rcpp::IntegerVector> axes) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = [&]() -> array {
    if (axes.isNotNull()) {
      Rcpp::IntegerVector axes_vec(axes.get());
      std::vector<int> ax(axes_vec.begin(), axes_vec.end());
      return squeeze(arr, normalize_axes(arr, ax));
    }
    return squeeze(arr);
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_expand_dims(SEXP xp_, Rcpp::IntegerVector axes_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  std::vector<int> axes(axes_.begin(), axes_.end());
  std::vector<int> normalized = normalize_new_axes(arr, axes);
  array result = expand_dims(arr, normalized);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_repeat(SEXP xp_, int repeats, Rcpp::Nullable<int> axis) {
  if (repeats <= 0) {
    Rcpp::stop("repeats must be positive.");
  }
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int ax = normalize_axis(arr, Rcpp::as<int>(axis.get()));
      return repeat(arr, repeats, ax);
    }
    return repeat(arr, repeats);
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_tile(SEXP xp_, Rcpp::IntegerVector reps_) {
  if (reps_.size() == 0) {
    Rcpp::stop("reps must contain at least one element.");
  }
  std::vector<int> reps(reps_.begin(), reps_.end());
  for (int value : reps) {
    if (value <= 0) {
      Rcpp::stop("All repetitions must be positive.");
    }
  }

  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = tile(arr, reps);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_roll(SEXP xp_, SEXP shift_, Rcpp::Nullable<Rcpp::IntegerVector> axes_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  NumericVector shifts(shift_);
  if (shifts.size() == 0) {
    Rcpp::stop("shift must contain at least one element.");
  }

  array result = [&]() -> array {
    if (axes_.isNotNull()) {
      Rcpp::IntegerVector axes_vec(axes_.get());
      if (axes_vec.size() != shifts.size()) {
        Rcpp::stop("shift and axis must have the same length.");
      }
      std::vector<int> axes(axes_vec.begin(), axes_vec.end());
      axes = normalize_axes(arr, axes);
      Shape shift_shape;
      shift_shape.reserve(shifts.size());
      for (double val : shifts) {
        shift_shape.push_back(static_cast<int>(val));
      }
      if (axes.size() == 1) {
        return roll(arr, static_cast<int>(shifts[0]), axes[0]);
      }
      return roll(arr, shift_shape, axes);
    }

    if (shifts.size() == 1) {
      return roll(arr, static_cast<int>(shifts[0]));
    }
    Shape shift_shape;
    shift_shape.reserve(shifts.size());
    for (double val : shifts) {
      shift_shape.push_back(static_cast<int>(val));
    }
    return roll(arr, shift_shape);
  }();

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_moveaxis(SEXP xp_, Rcpp::IntegerVector source_, Rcpp::IntegerVector destination_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  int ndim = static_cast<int>(arr.ndim());
  int n = source_.size();
  if (n == 0) {
    Rcpp::stop("source must contain at least one axis.");
  }
  if (destination_.size() != n) {
    Rcpp::stop("source and destination must have the same length.");
  }

  std::vector<int> source_norm;
  std::vector<int> dest_norm;
  source_norm.reserve(n);
  dest_norm.reserve(n);

  std::vector<bool> is_moved(ndim, false);
  for (int axis : source_) {
    int norm = normalize_axis(arr, axis);
    if (is_moved[norm]) {
      Rcpp::stop("source axes must be unique.");
    }
    is_moved[norm] = true;
    source_norm.push_back(norm);
  }

  std::vector<bool> dest_seen(ndim, false);
  for (int axis : destination_) {
    int norm = normalize_axis(arr, axis);
    if (dest_seen[norm]) {
      Rcpp::stop("destination axes must be unique.");
    }
    dest_seen[norm] = true;
    dest_norm.push_back(norm);
  }

  std::vector<std::pair<int, int>> moves;
  moves.reserve(n);
  for (int i = 0; i < n; ++i) {
    moves.emplace_back(dest_norm[i], source_norm[i]);
  }
  std::sort(
      moves.begin(),
      moves.end(),
      [](const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) {
        return lhs.first < rhs.first;
      });

  std::vector<int> remaining;
  remaining.reserve(ndim - n);
  for (int axis = 0; axis < ndim; ++axis) {
    if (!is_moved[axis]) {
      remaining.push_back(axis);
    }
  }

  std::vector<int> permutation;
  permutation.reserve(ndim);
  std::size_t move_idx = 0;
  std::size_t rem_idx = 0;
  for (int pos = 0; pos < ndim; ++pos) {
    if (move_idx < moves.size() && moves[move_idx].first == pos) {
      permutation.push_back(moves[move_idx].second);
      ++move_idx;
    } else {
      if (rem_idx >= remaining.size()) {
        Rcpp::stop("Invalid moveaxis configuration.");
      }
      permutation.push_back(remaining[rem_idx]);
      ++rem_idx;
    }
  }

  array result = transpose(arr, permutation);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_pad(SEXP xp_,
                 Rcpp::IntegerMatrix pad_pairs_,
                 double pad_value,
                 std::string dtype_str,
                 std::string device_str,
                 std::string mode_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  if (pad_pairs_.ncol() != 2) {
    Rcpp::stop("pad_width must have two columns (before, after).");
  }
  int ndim = static_cast<int>(arr.ndim());
  if (pad_pairs_.nrow() != ndim) {
    Rcpp::stop("pad_width row count must match tensor rank.");
  }

  std::vector<std::pair<int, int>> pad_width;
  pad_width.reserve(ndim);
  for (int i = 0; i < pad_pairs_.nrow(); ++i) {
    int before = pad_pairs_(i, 0);
    int after = pad_pairs_(i, 1);
    if (before < 0 || after < 0) {
      Rcpp::stop("pad widths must be non-negative.");
    }
    pad_width.emplace_back(before, after);
  }

  Dtype dtype = string_to_dtype(dtype_str);
  StreamOrDevice dev = string_to_device(device_str);
  array pad_val = array(pad_value, dtype);

  array result = pad(arr, pad_width, pad_val, mode_str, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_split(SEXP xp_,
                   Rcpp::Nullable<int> num_splits_,
                   Rcpp::Nullable<Rcpp::IntegerVector> indices_,
                   int axis,
                   std::string dtype_str,
                   std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  int ax = normalize_axis(arr, axis);
  StreamOrDevice dev = string_to_device(device_str);

  std::vector<array> outputs;

  if (indices_.isNotNull()) {
    Rcpp::IntegerVector indices_vec(indices_.get());
    if (indices_vec.size() == 0) {
      Rcpp::stop("indices must contain at least one value.");
    }
    Shape indices_shape;
    indices_shape.reserve(indices_vec.size());
    for (int value : indices_vec) {
      if (value < 0) {
        Rcpp::stop("Split indices must be non-negative.");
      }
      indices_shape.push_back(value);
    }
    outputs = split(arr, indices_shape, ax, dev);
  } else if (num_splits_.isNotNull()) {
    int num_splits = Rcpp::as<int>(num_splits_.get());
    if (num_splits <= 0) {
      Rcpp::stop("num_splits must be positive.");
    }
    outputs = split(arr, num_splits, ax, dev);
  } else {
    Rcpp::stop("Either num_splits or indices must be supplied.");
  }

  List out(outputs.size());
  for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
    out[i] = make_mlx_xptr(std::move(outputs[i]));
  }
  return out;
}

// [[Rcpp::export]]
SEXP cpp_mlx_where(SEXP cond_xp_, SEXP xp_true_, SEXP xp_false_,
                   std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* cond_wrapper = get_mlx_wrapper(cond_xp_);
  MlxArrayWrapper* true_wrapper = get_mlx_wrapper(xp_true_);
  MlxArrayWrapper* false_wrapper = get_mlx_wrapper(xp_false_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);

  array cond = astype(cond_wrapper->get(), bool_, target_device);
  array x = astype(true_wrapper->get(), target_dtype, target_device);
  array y = astype(false_wrapper->get(), target_dtype, target_device);

  array result = where(cond, x, y, target_device);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_take(SEXP xp_, SEXP indices_, int axis) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  IntegerVector idx(indices_);
  std::vector<int64_t> data(idx.begin(), idx.end());
  Shape shape{static_cast<int>(data.size())};
  array idx_array(data.data(), shape, int64);

  array result = take(arr, idx_array, axis);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_uniform(SEXP dim_, double low, double high,
                            std::string dtype_str, std::string device_str) {
  IntegerVector dim(dim_);
  Shape shape(dim.begin(), dim.end());

  Dtype dtype = string_to_dtype(dtype_str);
  if (dtype != float32 && dtype != float64) {
    Rcpp::stop("Random uniform currently supports dtype = \"float32\" or \"float64\" only.");
  }

  StreamOrDevice dev_input = string_to_device(device_str);
  array result = mlx::core::random::uniform(low, high, shape, dtype, std::nullopt, dev_input);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
void cpp_mlx_synchronize(std::string device_str) {
  Device dev = string_to_device(device_str);
  Stream stream = default_stream(dev);
  synchronize(stream);
}

// Indexing/slicing
// [[Rcpp::export]]
SEXP cpp_mlx_slice(SEXP xp_, SEXP starts_, SEXP stops_, SEXP strides_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  IntegerVector starts(starts_);
  IntegerVector stops(stops_);
  IntegerVector strides(strides_);

  // Convert to Shape
  Shape start_shape(starts.begin(), starts.end());
  Shape stop_shape(stops.begin(), stops.end());
  Shape stride_shape(strides.begin(), strides.end());

  array result = slice(wrapper->get(), start_shape, stop_shape, stride_shape);

  return make_mlx_xptr(std::move(result));
}

// Cumulative operations (flatten array)
// [[Rcpp::export]]
SEXP cpp_mlx_cumulative(SEXP xp_, std::string op) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  array arr = wrapper->get();

  array flat = [&]() -> array {
    if (arr.ndim() <= 1) {
      return reshape(arr, Shape{static_cast<int>(arr.size())});
    }

    std::vector<int> perm(arr.ndim());
    std::iota(perm.begin(), perm.end(), 0);
    std::reverse(perm.begin(), perm.end());

    array transposed = transpose(arr, perm);
    transposed = contiguous(transposed);
    return reshape(transposed, Shape{static_cast<int>(arr.size())});
  }();

  array result = [&]() -> array {
    if (op == "cumsum") {
      return cumsum(flat);
    } else if (op == "cumprod") {
      return cumprod(flat);
    } else if (op == "cummax") {
      return cummax(flat);
    } else if (op == "cummin") {
      return cummin(flat);
    } else {
      Rcpp::stop("Unsupported cumulative operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// Linear algebra operations
// [[Rcpp::export]]
SEXP cpp_mlx_solve(SEXP a_xp_, SEXP b_xp_,
                   std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);
  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);

  array result = [&]() -> array {
    if (b_xp_ == R_NilValue) {
      // No b provided: compute matrix inverse
      return linalg::inv(a_cpu, cpu_stream);
    } else {
      // b provided: solve linear system Ax = b
      MlxArrayWrapper* b_wrapper = get_mlx_wrapper(b_xp_);
      array b_cpu = astype(b_wrapper->get(), target_dtype, cpu_stream);
      return linalg::solve(a_cpu, b_cpu, cpu_stream);
    }
  }();

array result_target = astype(result, target_dtype, target_device);

  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_fft(SEXP xp_, bool inverse, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);

  array input_cpu = astype(wrapper->get(), wrapper->get().dtype(), cpu_stream);
  array result_cpu = inverse ? mlx::core::fft::ifftn(input_cpu, cpu_stream)
                             : mlx::core::fft::fftn(input_cpu, cpu_stream);

  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);

  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_cholesky(SEXP a_xp_, bool upper,
                      std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);

  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);
  array chol_cpu = mlx::core::linalg::cholesky(a_cpu, upper, cpu_stream);
  array chol_target = astype(chol_cpu, target_dtype, target_device);

  return make_mlx_xptr(std::move(chol_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_qr(SEXP a_xp_,
                std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);

  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);
  auto qr_cpu = mlx::core::linalg::qr(a_cpu, cpu_stream);

  array q_target = astype(qr_cpu.first, target_dtype, target_device);
  array r_target = astype(qr_cpu.second, target_dtype, target_device);

  return List::create(
      Named("Q") = make_mlx_xptr(std::move(q_target)),
      Named("R") = make_mlx_xptr(std::move(r_target)));
}

// [[Rcpp::export]]
SEXP cpp_mlx_svd(SEXP a_xp_, bool compute_uv,
                 std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);

  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);
  std::vector<array> svd_cpu = mlx::core::linalg::svd(a_cpu, compute_uv, cpu_stream);

  List out(svd_cpu.size());
  CharacterVector names(svd_cpu.size());
  if (svd_cpu.size() == 3) {
    names[0] = "U";
    names[1] = "S";
    names[2] = "Vh";
  } else if (svd_cpu.size() == 2) {
    names[0] = "S";
    names[1] = "Vh";
  } else if (svd_cpu.size() == 1) {
    names[0] = "S";
  }

  for (size_t i = 0; i < svd_cpu.size(); ++i) {
    array target = astype(svd_cpu[i], target_dtype, target_device);
    out[i] = make_mlx_xptr(std::move(target));
  }

  if (svd_cpu.size() > 0) {
    out.attr("names") = names;
  }

  return out;
}

// [[Rcpp::export]]
SEXP cpp_mlx_pinv(SEXP a_xp_,
                  std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);

  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);
  array pinv_cpu = mlx::core::linalg::pinv(a_cpu, cpu_stream);
  array pinv_target = astype(pinv_cpu, target_dtype, target_device);

  return make_mlx_xptr(std::move(pinv_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_norm(SEXP xp_, SEXP ord_,
                  Rcpp::Nullable<Rcpp::IntegerVector> axes,
                  bool keepdims, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array arr_cpu = astype(arr, arr.dtype(), cpu_stream);
  std::optional<std::vector<int>> axes_opt = optional_axes(arr, axes);

  array result_cpu = [&]() -> array {
    if (Rf_isNull(ord_)) {
      if (axes_opt.has_value()) {
        return mlx::core::linalg::norm(arr_cpu, axes_opt.value(), keepdims, cpu_stream);
      }
      return mlx::core::linalg::norm(arr_cpu, std::nullopt, keepdims, cpu_stream);
    }
    if (Rf_isReal(ord_) || Rf_isInteger(ord_)) {
      double ord_val = Rcpp::as<double>(ord_);
      if (axes_opt.has_value()) {
        return mlx::core::linalg::norm(arr_cpu, ord_val, axes_opt.value(), keepdims, cpu_stream);
      }
      return mlx::core::linalg::norm(arr_cpu, ord_val, std::nullopt, keepdims, cpu_stream);
    }
    if (Rf_isString(ord_)) {
      std::string ord_str = Rcpp::as<std::string>(ord_);
      if (axes_opt.has_value()) {
        return mlx::core::linalg::norm(arr_cpu, ord_str, axes_opt.value(), keepdims, cpu_stream);
      }
      return mlx::core::linalg::norm(arr_cpu, ord_str, std::nullopt, keepdims, cpu_stream);
    }
    Rcpp::stop("Unsupported ord type for mlx_norm.");
  }();

  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_eig(SEXP xp_, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array arr_cpu = astype(arr, arr.dtype(), cpu_stream);
  auto eig_pair = mlx::core::linalg::eig(arr_cpu, cpu_stream);

  array values_target = astype(eig_pair.first, eig_pair.first.dtype(), target_device);
  array vectors_target = astype(eig_pair.second, eig_pair.second.dtype(), target_device);

  return List::create(
      Named("values") = make_mlx_xptr(std::move(values_target)),
      Named("vectors") = make_mlx_xptr(std::move(vectors_target)));
}

// [[Rcpp::export]]
SEXP cpp_mlx_eigvals(SEXP xp_, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array arr_cpu = astype(arr, arr.dtype(), cpu_stream);
  array vals_cpu = mlx::core::linalg::eigvals(arr_cpu, cpu_stream);
  array vals_target = astype(vals_cpu, vals_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(vals_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_eigvalsh(SEXP xp_, std::string uplo, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array arr_cpu = astype(arr, arr.dtype(), cpu_stream);
  array vals_cpu = mlx::core::linalg::eigvalsh(arr_cpu, uplo, cpu_stream);
  array vals_target = astype(vals_cpu, vals_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(vals_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_eigh(SEXP xp_, std::string uplo, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array arr_cpu = astype(arr, arr.dtype(), cpu_stream);
  auto eig_pair = mlx::core::linalg::eigh(arr_cpu, uplo, cpu_stream);

  array values_target = astype(eig_pair.first, eig_pair.first.dtype(), target_device);
  array vectors_target = astype(eig_pair.second, eig_pair.second.dtype(), target_device);

  return List::create(
      Named("values") = make_mlx_xptr(std::move(values_target)),
      Named("vectors") = make_mlx_xptr(std::move(vectors_target)));
}

// [[Rcpp::export]]
SEXP cpp_mlx_solve_triangular(SEXP a_xp_, SEXP b_xp_, bool upper,
                              std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);
  MlxArrayWrapper* b_wrapper = get_mlx_wrapper(b_xp_);

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array a_cpu = astype(a_wrapper->get(), a_wrapper->get().dtype(), cpu_stream);
  array b_cpu = astype(b_wrapper->get(), b_wrapper->get().dtype(), cpu_stream);

  array result_cpu = mlx::core::linalg::solve_triangular(a_cpu, b_cpu, upper, cpu_stream);
  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_cross(SEXP a_xp_, SEXP b_xp_, int axis, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);
  MlxArrayWrapper* b_wrapper = get_mlx_wrapper(b_xp_);

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array a_cpu = astype(a_wrapper->get(), a_wrapper->get().dtype(), cpu_stream);
  array b_cpu = astype(b_wrapper->get(), b_wrapper->get().dtype(), cpu_stream);

  int axis_input = axis;
  if (axis_input >= 0) {
    axis_input -= 1;
  }
  int ax = normalize_axis(a_cpu, axis_input);
  array result_cpu = mlx::core::linalg::cross(a_cpu, b_cpu, ax, cpu_stream);
  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(result_target));
}
