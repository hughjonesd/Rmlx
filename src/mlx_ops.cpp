#include "mlx_bindings.hpp"
#include <mlx/mlx.h>
#include <Rcpp.h>
#include <string>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

// Unary operations
// [[Rcpp::export]]
SEXP cpp_mlx_unary(SEXP xp_, std::string op) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  array result = [&]() -> array {
    if (op == "neg") {
      return negative(wrapper->get());
    } else if (op == "abs") {
      return abs(wrapper->get());
    } else if (op == "sqrt") {
      return sqrt(wrapper->get());
    } else if (op == "exp") {
      return exp(wrapper->get());
    } else if (op == "log") {
      return log(wrapper->get());
    } else if (op == "sin") {
      return sin(wrapper->get());
    } else if (op == "cos") {
      return cos(wrapper->get());
    } else if (op == "tan") {
      return tan(wrapper->get());
    } else {
      Rcpp::stop("Unsupported unary operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// Binary operations
// [[Rcpp::export]]
SEXP cpp_mlx_binary(SEXP xp1_, SEXP xp2_, std::string op) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  array result = [&]() -> array {
    if (op == "+") {
      return add(wrapper1->get(), wrapper2->get());
    } else if (op == "-") {
      return subtract(wrapper1->get(), wrapper2->get());
    } else if (op == "*") {
      return multiply(wrapper1->get(), wrapper2->get());
    } else if (op == "/") {
      return divide(wrapper1->get(), wrapper2->get());
    } else if (op == "^") {
      return power(wrapper1->get(), wrapper2->get());
    } else if (op == "==") {
      return equal(wrapper1->get(), wrapper2->get());
    } else if (op == "!=") {
      return not_equal(wrapper1->get(), wrapper2->get());
    } else if (op == "<") {
      return less(wrapper1->get(), wrapper2->get());
    } else if (op == "<=") {
      return less_equal(wrapper1->get(), wrapper2->get());
    } else if (op == ">") {
      return greater(wrapper1->get(), wrapper2->get());
    } else if (op == ">=") {
      return greater_equal(wrapper1->get(), wrapper2->get());
    } else {
      Rcpp::stop("Unsupported binary operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// Reductions
// [[Rcpp::export]]
SEXP cpp_mlx_reduce(SEXP xp_, std::string op) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  array result = [&]() -> array {
    if (op == "sum") {
      return sum(wrapper->get(), false);  // keepdims=false
    } else if (op == "mean") {
      return mean(wrapper->get(), false);
    } else if (op == "min") {
      return min(wrapper->get(), false);
    } else if (op == "max") {
      return max(wrapper->get(), false);
    } else {
      Rcpp::stop("Unsupported reduction operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// Axis reductions
// [[Rcpp::export]]
SEXP cpp_mlx_reduce_axis(SEXP xp_, std::string op, int axis, bool keepdims) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  std::vector<int> axes = {axis};

  array result = [&]() -> array {
    if (op == "sum") {
      return sum(wrapper->get(), axes, keepdims);
    } else if (op == "mean") {
      return mean(wrapper->get(), axes, keepdims);
    } else if (op == "min") {
      return min(wrapper->get(), axes, keepdims);
    } else if (op == "max") {
      return max(wrapper->get(), axes, keepdims);
    } else {
      Rcpp::stop("Unsupported axis reduction operation: " + op);
    }
  }();

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
SEXP cpp_mlx_matmul(SEXP xp1_, SEXP xp2_) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  array result = matmul(wrapper1->get(), wrapper2->get());

  return make_mlx_xptr(std::move(result));
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
