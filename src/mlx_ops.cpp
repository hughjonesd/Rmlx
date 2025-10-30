// Core MLX operations
#include "mlx_helpers.hpp"
#include <mlx/mlx.h>
#include <mlx/fft.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

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
SEXP cpp_mlx_addmm(SEXP input_xp_,
                   SEXP mat1_xp_,
                   SEXP mat2_xp_,
                   double alpha,
                   double beta,
                   std::string dtype_str,
                   std::string device_str) {
  MlxArrayWrapper* input_wrapper = get_mlx_wrapper(input_xp_);
  MlxArrayWrapper* mat1_wrapper = get_mlx_wrapper(mat1_xp_);
  MlxArrayWrapper* mat2_wrapper = get_mlx_wrapper(mat2_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);

  array input_arr = astype(input_wrapper->get(), target_dtype, target_device);
  array mat1_arr = astype(mat1_wrapper->get(), target_dtype, target_device);
  array mat2_arr = astype(mat2_wrapper->get(), target_dtype, target_device);

  array result = addmm(
    std::move(input_arr),
    std::move(mat1_arr),
    std::move(mat2_arr),
    static_cast<float>(alpha),
    static_cast<float>(beta),
    target_device);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_hadamard_transform(SEXP xp_,
                                Rcpp::Nullable<double> scale_,
                                std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice target_device = string_to_device(device_str);
  arr = astype(arr, arr.dtype(), target_device);

  std::optional<float> scale = std::nullopt;
  if (scale_.isNotNull()) {
    scale = static_cast<float>(Rcpp::as<double>(scale_));
  }

  array result = hadamard_transform(arr, scale, target_device);
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
SEXP cpp_mlx_tril(SEXP xp_, int k, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice dev = string_to_device(device_str);
  array result = tril(arr, k, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_triu(SEXP xp_, int k, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice dev = string_to_device(device_str);
  array result = triu(arr, k, dev);
  return make_mlx_xptr(std::move(result));
}
