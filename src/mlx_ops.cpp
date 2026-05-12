// Core MLX operations
#include "mlx_helpers.hpp"
#include "colmajor_helpers.hpp"
#include <mlx/mlx.h>
#include <mlx/fft.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

// [[Rcpp::export]]
SEXP cpp_mlx_matmul(SEXP xp1_, SEXP xp2_,
                    std::string dtype_str) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  Dtype target_dtype = string_to_dtype(dtype_str);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  lhs = astype(lhs, target_dtype);
  rhs = astype(rhs, target_dtype);

  array result = matmul(lhs, rhs);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_addmm(SEXP input_xp_,
                   SEXP mat1_xp_,
                   SEXP mat2_xp_,
                   double alpha,
                   double beta,
                   std::string dtype_str) {
  MlxArrayWrapper* input_wrapper = get_mlx_wrapper(input_xp_);
  MlxArrayWrapper* mat1_wrapper = get_mlx_wrapper(mat1_xp_);
  MlxArrayWrapper* mat2_wrapper = get_mlx_wrapper(mat2_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);

  array input_arr = astype(input_wrapper->get(), target_dtype);
  array mat1_arr = astype(mat1_wrapper->get(), target_dtype);
  array mat2_arr = astype(mat2_wrapper->get(), target_dtype);

  array result = addmm(
    std::move(input_arr),
    std::move(mat1_arr),
    std::move(mat2_arr),
    static_cast<float>(alpha),
    static_cast<float>(beta));

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_hadamard_transform(SEXP xp_,
                                Rcpp::Nullable<double> scale_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  std::optional<float> scale = std::nullopt;
  if (scale_.isNotNull()) {
    scale = static_cast<float>(Rcpp::as<double>(scale_));
  }

  array result = hadamard_transform(arr, scale);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_cast(SEXP xp_, std::string dtype_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  Dtype dtype = string_to_dtype(dtype_str);

  array arr = wrapper->get();
  array result = astype(arr, dtype);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_cumulative(SEXP xp_, std::string op) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  array arr = wrapper->get();

  array flat = flatten_r_order(arr);

  array result = [&]() -> array {
    if (op == "cumsum") {
      return cumsum(flat, false, true);
    } else if (op == "cumprod") {
      return cumprod(flat, false, true);
    } else if (op == "cummax") {
      return cummax(flat, false, true);
    } else if (op == "cummin") {
      return cummin(flat, false, true);
    } else {
      Rcpp::stop("Unsupported cumulative operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_fft(SEXP xp_,
                 Rcpp::Nullable<Rcpp::IntegerVector> axes_,
                 bool inverse) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  array input = wrapper->get();

  array result = [&]() -> array {
    if (axes_.isNull()) {
#if MLX_VERSION_NUMERIC >= 31002
      return inverse
        ? mlx::core::fft::ifftn(input, mlx::core::fft::FFTNorm::Backward)
        : mlx::core::fft::fftn(input, mlx::core::fft::FFTNorm::Backward);
#else
      return inverse ? mlx::core::fft::ifftn(input)
                     : mlx::core::fft::fftn(input);
#endif
    }
    std::vector<int> axes = Rcpp::as<std::vector<int>>(axes_.get());
#if MLX_VERSION_NUMERIC >= 31002
    return inverse
      ? mlx::core::fft::ifftn(input, axes, mlx::core::fft::FFTNorm::Backward)
      : mlx::core::fft::fftn(input, axes, mlx::core::fft::FFTNorm::Backward);
#else
    return inverse ? mlx::core::fft::ifftn(input, axes)
                   : mlx::core::fft::fftn(input, axes);
#endif
  }();

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_cumsum(SEXP xp_, Rcpp::Nullable<int> axis_, bool reverse, bool inclusive) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = axis_.isNull()
    ? cumsum(arr, reverse, inclusive)
    : cumsum(arr, Rcpp::as<int>(axis_), reverse, inclusive);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_cumprod(SEXP xp_, Rcpp::Nullable<int> axis_, bool reverse, bool inclusive) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = axis_.isNull()
    ? cumprod(arr, reverse, inclusive)
    : cumprod(arr, Rcpp::as<int>(axis_), reverse, inclusive);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
void cpp_mlx_synchronize(std::string device_str) {
  Device dev = string_to_device(device_str);
  Stream stream = default_stream(dev);
  synchronize(stream);
}

// [[Rcpp::export]]
SEXP cpp_mlx_tril(SEXP xp_, int k) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = tril(arr, k);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_triu(SEXP xp_, int k) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = triu(arr, k);
  return make_mlx_xptr(std::move(result));
}
