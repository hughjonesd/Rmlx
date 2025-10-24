// Random number generation
#include "mlx_helpers.hpp"
#include <mlx/mlx.h>
#include <mlx/random.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

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

  Dtype dtype = string_to_dtype(dtype_str);
  if (dtype != float32 && dtype != float64) {
    Rcpp::stop("Random gumbel currently supports dtype = \"float32\" or \"float64\" only.");
  }
  StreamOrDevice dev_input = string_to_device(device_str);
  array result = mlx::core::random::gumbel(shape, dtype, std::nullopt, dev_input);
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

  Dtype dtype = string_to_dtype(dtype_str);
  if (dtype != float32 && dtype != float64) {
    Rcpp::stop("Random truncated_normal currently supports dtype = \"float32\" or \"float64\" only.");
  }
  StreamOrDevice dev_input = string_to_device(device_str);
  array lower_arr(static_cast<float>(lower), dtype);
  array upper_arr(static_cast<float>(upper), dtype);
  array result = mlx::core::random::truncated_normal(lower_arr, upper_arr, shape, dtype, std::nullopt, dev_input);
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

  Dtype dtype = string_to_dtype(dtype_str);
  if (dtype != float32 && dtype != float64) {
    Rcpp::stop("Random multivariate_normal currently supports dtype = \"float32\" or \"float64\" only.");
  }
  StreamOrDevice dev_input = string_to_device(device_str);
  array result = mlx::core::random::multivariate_normal(mean_arr, cov_arr, shape, dtype, std::nullopt, dev_input);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_laplace(SEXP dim_, double loc, double scale,
                            std::string dtype_str, std::string device_str) {
  IntegerVector dim(dim_);
  Shape shape(dim.begin(), dim.end());

  Dtype dtype = string_to_dtype(dtype_str);
  if (dtype != float32 && dtype != float64) {
    Rcpp::stop("Random laplace currently supports dtype = \"float32\" or \"float64\" only.");
  }
  StreamOrDevice dev_input = string_to_device(device_str);
  array result = mlx::core::random::laplace(shape, dtype, static_cast<float>(loc), static_cast<float>(scale), std::nullopt, dev_input);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_categorical(SEXP logits_, int axis, int num_samples) {
  List logits_obj(logits_);
  array logits_arr = get_mlx_wrapper(logits_obj["ptr"])->get();
  std::string device_str = Rcpp::as<std::string>(logits_obj["device"]);

  StreamOrDevice dev = string_to_device(device_str);
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
  array x_arr = get_mlx_wrapper(x_obj["ptr"])->get();
  std::string device_str = Rcpp::as<std::string>(x_obj["device"]);

  StreamOrDevice dev = string_to_device(device_str);
  array result = mlx::core::random::permutation(x_arr, axis, std::nullopt, dev);
  return make_mlx_xptr(std::move(result));
}

