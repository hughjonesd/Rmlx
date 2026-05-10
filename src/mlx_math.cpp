// Math operations (unary, binary, logical)
#include "mlx_helpers.hpp"
#include <mlx/mlx.h>
#include <optional>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

// [[Rcpp::export]]
SEXP cpp_mlx_unary(SEXP xp_, std::string op) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array input = wrapper->get();
  StreamOrDevice dev = wrapper->stream(input.dtype());

  array result = [&]() -> array {
    if (op == "neg") {
      return negative(input, dev);
    } else if (op == "abs") {
      return abs(input, dev);
    } else if (op == "sign") {
      return sign(input, dev);
    } else if (op == "sqrt") {
      return sqrt(input, dev);
    } else if (op == "rsqrt") {
      return rsqrt(input, dev);
    } else if (op == "square") {
      return square(input, dev);
    } else if (op == "exp") {
      return exp(input, dev);
    } else if (op == "expm1") {
      return expm1(input, dev);
    } else if (op == "log") {
      return log(input, dev);
    } else if (op == "log2") {
      return log2(input, dev);
    } else if (op == "log10") {
      return log10(input, dev);
    } else if (op == "log1p") {
      return log1p(input, dev);
    } else if (op == "sin") {
      return sin(input, dev);
    } else if (op == "cos") {
      return cos(input, dev);
    } else if (op == "tan") {
      return tan(input, dev);
    } else if (op == "asin") {
      return arcsin(input, dev);
    } else if (op == "acos") {
      return arccos(input, dev);
    } else if (op == "atan") {
      return arctan(input, dev);
    } else if (op == "sinh") {
      return sinh(input, dev);
    } else if (op == "cosh") {
      return cosh(input, dev);
    } else if (op == "tanh") {
      return tanh(input, dev);
    } else if (op == "asinh") {
      return arcsinh(input, dev);
    } else if (op == "acosh") {
      return arccosh(input, dev);
    } else if (op == "atanh") {
      return arctanh(input, dev);
    } else if (op == "erf") {
      return erf(input, dev);
    } else if (op == "erfinv") {
      return erfinv(input, dev);
    } else if (op == "floor") {
      return floor(input, dev);
    } else if (op == "ceil") {
      return ceil(input, dev);
    } else if (op == "round") {
      return round(input, dev);
    } else if (op == "isnan") {
      return isnan(input, dev);
    } else if (op == "isinf") {
      return isinf(input, dev);
    } else if (op == "isfinite") {
      return isfinite(input, dev);
    } else if (op == "isposinf") {
      return isposinf(input, dev);
    } else if (op == "isneginf") {
      return isneginf(input, dev);
    } else if (op == "real") {
      return real(input, dev);
    } else if (op == "imag") {
      return imag(input, dev);
    } else if (op == "conj") {
      return conjugate(input, dev);
    } else if (op == "degrees") {
      return degrees(input, dev);
    } else if (op == "radians") {
      return radians(input, dev);
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
  StreamOrDevice dev = wrapper->stream(bool_);
  array arr_bool = astype(arr, bool_, dev);
  array result = logical_not(arr_bool, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_binary(SEXP xp1_, SEXP xp2_, std::string op,
                    std::string dtype_str) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = wrapper1->stream(target_dtype);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  lhs = astype(lhs, target_dtype, target_device);
  rhs = astype(rhs, target_dtype, target_device);

  array result = [&]() -> array {
    if (op == "+") {
      return add(lhs, rhs, target_device);
    } else if (op == "-") {
      return subtract(lhs, rhs, target_device);
    } else if (op == "*") {
      return multiply(lhs, rhs, target_device);
    } else if (op == "/") {
      return divide(lhs, rhs, target_device);
    } else if (op == "^") {
      return power(lhs, rhs, target_device);
    } else if (op == "==") {
      return equal(lhs, rhs, target_device);
    } else if (op == "!=") {
      return not_equal(lhs, rhs, target_device);
    } else if (op == "<") {
      return less(lhs, rhs, target_device);
    } else if (op == "<=") {
      return less_equal(lhs, rhs, target_device);
    } else if (op == ">") {
      return greater(lhs, rhs, target_device);
    } else if (op == ">=") {
      return greater_equal(lhs, rhs, target_device);
    } else {
      Rcpp::stop("Unsupported binary operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_minimum(SEXP xp1_, SEXP xp2_) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  Dtype target_dtype = lhs.dtype();
  if (target_dtype == bool_) {
    target_dtype = float32;
  }
  if (rhs.dtype() == float64 || target_dtype == float64) {
    target_dtype = float64;
  } else if (rhs.dtype() == float32 || target_dtype == float32) {
    target_dtype = float32;
  }
  StreamOrDevice target_device = wrapper1->stream(target_dtype);

  lhs = astype(lhs, target_dtype, target_device);
  rhs = astype(rhs, target_dtype, target_device);

  array result = minimum(lhs, rhs, target_device);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_maximum(SEXP xp1_, SEXP xp2_) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  Dtype target_dtype = lhs.dtype();
  if (target_dtype == bool_) {
    target_dtype = float32;
  }
  if (rhs.dtype() == float64 || target_dtype == float64) {
    target_dtype = float64;
  } else if (rhs.dtype() == float32 || target_dtype == float32) {
    target_dtype = float32;
  }
  StreamOrDevice target_device = wrapper1->stream(target_dtype);

  lhs = astype(lhs, target_dtype, target_device);
  rhs = astype(rhs, target_dtype, target_device);

  array result = maximum(lhs, rhs, target_device);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_clip(SEXP xp_, SEXP min_, SEXP max_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  Dtype original_dtype = arr.dtype();
  StreamOrDevice target_device = wrapper->stream(original_dtype);

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
SEXP cpp_mlx_floor_divide(SEXP xp1_, SEXP xp2_) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  Dtype target_dtype = promote_numeric_dtype(lhs.dtype(), rhs.dtype());
  StreamOrDevice target_device = wrapper1->stream(target_dtype);

  lhs = astype(lhs, target_dtype, target_device);
  rhs = astype(rhs, target_dtype, target_device);

  array result = floor_divide(lhs, rhs, target_device);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_remainder(SEXP xp1_, SEXP xp2_) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  Dtype target_dtype = promote_numeric_dtype(lhs.dtype(), rhs.dtype());
  StreamOrDevice target_device = wrapper1->stream(target_dtype);

  lhs = astype(lhs, target_dtype, target_device);
  rhs = astype(rhs, target_dtype, target_device);

  array result = remainder(lhs, rhs, target_device);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_logical(SEXP xp1_, SEXP xp2_, std::string op) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  StreamOrDevice target_device = wrapper1->stream(bool_);

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

// [[Rcpp::export]]
SEXP cpp_mlx_isclose(SEXP xp1_, SEXP xp2_, double rtol, double atol, bool equal_nan) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  StreamOrDevice target_device = wrapper1->stream(wrapper1->get().dtype());

  array lhs = astype(wrapper1->get(), wrapper1->get().dtype(), target_device);
  array rhs = astype(wrapper2->get(), wrapper2->get().dtype(), target_device);

  array result = isclose(lhs, rhs, rtol, atol, equal_nan, target_device);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_allclose(SEXP xp1_, SEXP xp2_, double rtol, double atol, bool equal_nan) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  StreamOrDevice target_device = wrapper1->stream(wrapper1->get().dtype());

  array lhs = astype(wrapper1->get(), wrapper1->get().dtype(), target_device);
  array rhs = astype(wrapper2->get(), wrapper2->get().dtype(), target_device);

  array result = allclose(lhs, rhs, rtol, atol, equal_nan, target_device);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_nan_to_num(SEXP xp_,
                        Rcpp::Nullable<double> nan_,
                        Rcpp::Nullable<double> posinf_,
                        Rcpp::Nullable<double> neginf_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  float nan_value = nan_.isNull() ? 0.0f : static_cast<float>(Rcpp::as<double>(nan_.get()));

  std::optional<float> posinf_opt;
  if (posinf_.isNotNull()) {
    posinf_opt = static_cast<float>(Rcpp::as<double>(posinf_.get()));
  }

  std::optional<float> neginf_opt;
  if (neginf_.isNotNull()) {
    neginf_opt = static_cast<float>(Rcpp::as<double>(neginf_.get()));
  }

  array result = nan_to_num(arr, nan_value, posinf_opt, neginf_opt);
  return make_mlx_xptr(std::move(result));
}
