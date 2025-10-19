#include "mlx_bindings.hpp"
#include <mlx/mlx.h>
#include <Rcpp.h>
#include <vector>
#include <string>

using namespace Rcpp;
using namespace mlx::core;

namespace rmlx {

// MlxArrayWrapper implementation
MlxArrayWrapper::MlxArrayWrapper() : ptr_(nullptr) {}

MlxArrayWrapper::MlxArrayWrapper(const array& arr)
    : ptr_(std::make_shared<array>(arr)) {}

MlxArrayWrapper::MlxArrayWrapper(array&& arr)
    : ptr_(std::make_shared<array>(std::move(arr))) {}

// Finalizer for R external pointer
void mlx_array_finalizer(SEXP xp) {
  if (TYPEOF(xp) == EXTPTRSXP) {
    MlxArrayWrapper* wrapper = static_cast<MlxArrayWrapper*>(R_ExternalPtrAddr(xp));
    if (wrapper != nullptr) {
      delete wrapper;
      R_ClearExternalPtr(xp);
    }
  }
}

// Get MlxArrayWrapper from external pointer
MlxArrayWrapper* get_mlx_wrapper(SEXP xp) {
  if (TYPEOF(xp) != EXTPTRSXP) {
    Rcpp::stop("Expected external pointer");
  }
  MlxArrayWrapper* wrapper = static_cast<MlxArrayWrapper*>(R_ExternalPtrAddr(xp));
  if (wrapper == nullptr || wrapper->is_null()) {
    Rcpp::stop("Invalid MLX array pointer");
  }
  return wrapper;
}

// Wrap MLX array in external pointer
SEXP make_mlx_xptr(const array& arr) {
  MlxArrayWrapper* wrapper = new MlxArrayWrapper(arr);
  SEXP xp = R_MakeExternalPtr(wrapper, R_NilValue, R_NilValue);
  R_RegisterCFinalizerEx(xp, mlx_array_finalizer, TRUE);
  return xp;
}

SEXP make_mlx_xptr(array&& arr) {
  MlxArrayWrapper* wrapper = new MlxArrayWrapper(std::move(arr));
  SEXP xp = R_MakeExternalPtr(wrapper, R_NilValue, R_NilValue);
  R_RegisterCFinalizerEx(xp, mlx_array_finalizer, TRUE);
  return xp;
}

// Helper: convert dtype string to MLX dtype
Dtype string_to_dtype(const std::string& dtype) {
  if (dtype == "float32") return float32;
  if (dtype == "float64") return float64;
  if (dtype == "int32") return int32;
  if (dtype == "int64") return int64;
  if (dtype == "bool") return bool_;
  Rcpp::stop("Unsupported dtype: " + dtype);
}

// Helper: convert dtype to string
std::string dtype_to_string(Dtype dtype) {
  if (dtype == float32) return "float32";
  if (dtype == float64) return "float64";
  if (dtype == int32) return "int32";
  if (dtype == int64) return "int64";
  if (dtype == bool_) return "bool";
  return "unknown";
}

// Helper: get device
Device string_to_device(const std::string& device) {
  if (device == "gpu") return Device(Device::gpu);
  if (device == "cpu") return Device(Device::cpu);
  Rcpp::stop("Unsupported device: " + device);
}

} // namespace rmlx

using namespace rmlx;

// [[Rcpp::export]]
SEXP cpp_mlx_from_numeric(SEXP x_, SEXP dim_, SEXP dtype_, SEXP device_) {
  NumericVector x(x_);
  IntegerVector dim(dim_);
  std::string dtype_str = as<std::string>(dtype_);
  std::string device_str = as<std::string>(device_);

  // Convert dimensions - MLX uses Shape which is SmallVector<int>
  Shape shape(dim.begin(), dim.end());

  // Create MLX array
  Dtype dt = string_to_dtype(dtype_str);
  StreamOrDevice dev = string_to_device(device_str);

  // Copy data from R to MLX on CPU first, then move to device
  // R uses column-major, MLX uses row-major
  array arr_cpu = [&]() -> array {
    if (dt == float64) {
      return array(x.begin(), shape, float64);
    } else if (dt == float32) {
      std::vector<float> data_f32(x.begin(), x.end());
      return array(data_f32.begin(), shape, float32);
    } else {
      Rcpp::stop("Unsupported dtype for conversion from numeric");
    }
  }();

  // Move to target device if needed
  array arr = astype(arr_cpu, dt, dev);

  return make_mlx_xptr(std::move(arr));
}

// [[Rcpp::export]]
SEXP cpp_mlx_empty(SEXP dim_, SEXP dtype_, SEXP device_) {
  IntegerVector dim(dim_);
  std::string dtype_str = as<std::string>(dtype_);
  std::string device_str = as<std::string>(device_);

  Shape shape(dim.begin(), dim.end());
  Dtype dt = string_to_dtype(dtype_str);
  StreamOrDevice dev = string_to_device(device_str);

  array arr = zeros(shape, dt, dev);

  return make_mlx_xptr(std::move(arr));
}

// [[Rcpp::export]]
SEXP cpp_mlx_to_numeric(SEXP xp_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  // Evaluate first
  eval(arr);

  // Get total size
  int total_size = arr.size();

  // Create output vector
  NumericVector result(total_size);

  // Copy data
  if (arr.dtype() == float64) {
    const double* data = arr.data<double>();
    std::copy(data, data + total_size, result.begin());
  } else if (arr.dtype() == float32) {
    const float* data = arr.data<float>();
    for (int i = 0; i < total_size; ++i) {
      result[i] = static_cast<double>(data[i]);
    }
  } else {
    Rcpp::stop("Unsupported dtype for conversion to numeric");
  }

  return result;
}

// [[Rcpp::export]]
void cpp_mlx_eval(SEXP xp_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  eval(wrapper->get());
}

// [[Rcpp::export]]
IntegerVector cpp_mlx_shape(SEXP xp_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  const auto& shape = wrapper->get().shape();

  IntegerVector result(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    result[i] = shape[i];
  }

  return result;
}

// [[Rcpp::export]]
std::string cpp_mlx_dtype(SEXP xp_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  return dtype_to_string(wrapper->get().dtype());
}
