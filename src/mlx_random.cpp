// Random number generation
#include "mlx_helpers.hpp"
#include <mlx/mlx.h>
#include <mlx/random.h>
#include <mlx/ops.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

namespace {

Dtype checked_float_random_dtype(const std::string& dtype_str,
                                 const std::string& distribution) {
  Dtype dtype = string_to_dtype(dtype_str);
  if (dtype != float32 && dtype != float64) {
    Rcpp::stop(
      "Random " + distribution +
      " currently supports dtype = \"float32\" or \"float64\" only.");
  }
  return dtype;
}

Dtype random_kernel_dtype(Dtype dtype) {
  return dtype == float64 ? float32 : dtype;
}

array finish_float_random(array result, Dtype requested_dtype) {
  if (requested_dtype == float64) {
    return astype(result, float64, Device(Device::cpu));
  }
  return result;
}

} // namespace

// [[Rcpp::export]]
SEXP cpp_mlx_random_key(double seed) {
  uint64_t seed_val = static_cast<uint64_t>(seed);
  array result = mlx::core::random::key(seed_val);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_split(SEXP key_xp_, int num) {
  if (num <= 0) {
    Rcpp::stop("`num` must be a positive integer.");
  }

  MlxArrayWrapper* key_wrapper = get_mlx_wrapper(key_xp_);
  array key_arr = key_wrapper->get();

  std::vector<array> outputs;
  outputs.reserve(num);

  array split_result = mlx::core::random::split(key_arr, num);
  for (int i = 0; i < num; ++i) {
    array sub_key = take(split_result, i, 0);
    outputs.push_back(std::move(sub_key));
  }

  List result(num);
  for (int i = 0; i < num; ++i) {
    result[i] = make_mlx_xptr(std::move(outputs[i]));
  }
  return result;
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_bits(SEXP dim_, int width, SEXP key_xp_,
                         std::string device_str) {
  IntegerVector dim(dim_);
  if (dim.size() == 0) {
    Rcpp::stop("`dim` must contain at least one element.");
  }
  if (width <= 0) {
    Rcpp::stop("`width` must be a positive integer.");
  }

  Shape shape(dim.begin(), dim.end());
  StreamOrDevice dev = string_to_device(device_str);

  std::optional<array> key_opt = std::nullopt;
  if (!Rf_isNull(key_xp_)) {
    MlxArrayWrapper* key_wrapper = get_mlx_wrapper(key_xp_);
    key_opt = key_wrapper->get();
  }

  array result = mlx::core::random::bits(shape, width, key_opt, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_normal(SEXP dim_, double mean, double std,
                           std::string dtype_str, std::string device_str) {
  IntegerVector dim(dim_);
  Shape shape(dim.begin(), dim.end());

  Dtype dtype = checked_float_random_dtype(dtype_str, "normal");
  StreamOrDevice dev_input = typed_device(dtype, device_str);
  Dtype rng_dtype = random_kernel_dtype(dtype);
  array result = mlx::core::random::normal(
      shape,
      rng_dtype,
      array(static_cast<float>(mean), rng_dtype),
      array(static_cast<float>(std), rng_dtype),
      std::nullopt,
      dev_input);
  result = finish_float_random(std::move(result), dtype);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_uniform(SEXP dim_, double low, double high,
                            std::string dtype_str, std::string device_str) {
  IntegerVector dim(dim_);
  Shape shape(dim.begin(), dim.end());

  Dtype dtype = checked_float_random_dtype(dtype_str, "uniform");
  StreamOrDevice dev_input = typed_device(dtype, device_str);
  Dtype rng_dtype = random_kernel_dtype(dtype);
  array result = mlx::core::random::uniform(low, high, shape, rng_dtype, std::nullopt, dev_input);
  result = finish_float_random(std::move(result), dtype);
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
SEXP cpp_mlx_random_gumbel(SEXP dim_, std::string dtype_str,
                           std::string device_str) {
  IntegerVector dim(dim_);
  Shape shape(dim.begin(), dim.end());

  Dtype dtype = checked_float_random_dtype(dtype_str, "gumbel");
  StreamOrDevice dev_input = typed_device(dtype, device_str);
  Dtype rng_dtype = random_kernel_dtype(dtype);
  array result = mlx::core::random::gumbel(shape, rng_dtype, std::nullopt, dev_input);
  result = finish_float_random(std::move(result), dtype);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_truncated_normal(SEXP lower_, SEXP upper_, SEXP dim_,
                                     std::string dtype_str, std::string device_str) {
  double lower = Rcpp::as<double>(lower_);
  double upper = Rcpp::as<double>(upper_);
  IntegerVector dim(dim_);
  Shape shape(dim.begin(), dim.end());

  if (lower >= upper) {
    Rcpp::stop("lower must be less than upper.");
  }

  Dtype dtype = checked_float_random_dtype(dtype_str, "truncated_normal");
  StreamOrDevice dev_input = typed_device(dtype, device_str);
  Dtype rng_dtype = random_kernel_dtype(dtype);
  array lower_arr(static_cast<float>(lower), rng_dtype);
  array upper_arr(static_cast<float>(upper), rng_dtype);
  array result = mlx::core::random::truncated_normal(
      lower_arr, upper_arr, shape, rng_dtype, std::nullopt, dev_input);
  result = finish_float_random(std::move(result), dtype);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_multivariate_normal(SEXP mean_, SEXP cov_, SEXP dim_,
                                        std::string dtype_str, std::string device_str) {
  IntegerVector dim(dim_);
  Shape shape(dim.begin(), dim.end());

  List mean_obj(mean_);
  array mean_arr = get_mlx_wrapper(mean_obj["ptr"])->get();

  List cov_obj(cov_);
  array cov_arr = get_mlx_wrapper(cov_obj["ptr"])->get();

  Dtype dtype = checked_float_random_dtype(dtype_str, "multivariate_normal");
  StreamOrDevice dev_input = typed_device(dtype, device_str);
  Dtype rng_dtype = random_kernel_dtype(dtype);
  mean_arr = astype(mean_arr, rng_dtype, dev_input);
  cov_arr = astype(cov_arr, rng_dtype, dev_input);
  array result = mlx::core::random::multivariate_normal(
      mean_arr, cov_arr, shape, rng_dtype, std::nullopt, dev_input);
  result = finish_float_random(std::move(result), dtype);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_laplace(SEXP dim_, double loc, double scale,
                            std::string dtype_str, std::string device_str) {
  IntegerVector dim(dim_);
  Shape shape(dim.begin(), dim.end());

  Dtype dtype = checked_float_random_dtype(dtype_str, "laplace");
  StreamOrDevice dev_input = typed_device(dtype, device_str);
  Dtype rng_dtype = random_kernel_dtype(dtype);
  array result = mlx::core::random::laplace(
      shape, rng_dtype, static_cast<float>(loc), static_cast<float>(scale), std::nullopt, dev_input);
  result = finish_float_random(std::move(result), dtype);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_categorical(SEXP logits_, int axis, int num_samples) {
  List logits_obj(logits_);
  MlxArrayWrapper* logits_wrapper = get_mlx_wrapper(logits_obj["ptr"]);
  array logits_arr = logits_wrapper->get();

  StreamOrDevice dev = logits_wrapper->stream(logits_arr.dtype());
  array result = mlx::core::random::categorical(logits_arr, axis, num_samples, std::nullopt, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_randint(SEXP dim_, int low, int high,
                            std::string dtype_str, std::string device_str) {
  IntegerVector dim(dim_);
  Shape shape(dim.begin(), dim.end());

  Dtype dtype = string_to_dtype(dtype_str);
  StreamOrDevice dev_input = string_to_device(device_str);
  array result = mlx::core::random::randint(low, high, shape, dtype, std::nullopt, dev_input);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_permutation_n(int n, std::string device_str) {
  StreamOrDevice dev = string_to_device(device_str);
  array result = mlx::core::random::permutation(n, std::nullopt, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_permutation(SEXP x_, int axis) {
  List x_obj(x_);
  MlxArrayWrapper* x_wrapper = get_mlx_wrapper(x_obj["ptr"]);
  array x_arr = x_wrapper->get();

  StreamOrDevice dev = x_wrapper->stream(x_arr.dtype());
  array result = mlx::core::random::permutation(x_arr, axis, std::nullopt, dev);
  return make_mlx_xptr(std::move(result));
}
