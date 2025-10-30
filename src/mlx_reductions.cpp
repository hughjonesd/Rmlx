// Reduction operations
#include "mlx_helpers.hpp"
#include <mlx/mlx.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

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

// [[Rcpp::export]]
SEXP cpp_mlx_argmax(SEXP xp_, Rcpp::Nullable<int> axis, bool keepdims) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int ax = Rcpp::as<int>(axis.get());
      ax = normalize_axis(arr, ax);
      array idx = argmax(arr, ax, keepdims);
      idx = idx + 1;
      return idx;
    }
    array idx = argmax(arr, keepdims);
    idx = idx + 1;
    return idx;
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
      array idx = argmin(arr, ax, keepdims);
      idx = idx + 1;
      return idx;
    }
    array idx = argmin(arr, keepdims);
    idx = idx + 1;
    return idx;
  }();

  return make_mlx_xptr(std::move(result));
}

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
