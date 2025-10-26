// Array creation functions
#include "mlx_helpers.hpp"
#include <mlx/mlx.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

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

