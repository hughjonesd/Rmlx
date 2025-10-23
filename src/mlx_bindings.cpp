#include "mlx_bindings.hpp"
#include <mlx/mlx.h>
#include <Rcpp.h>
#include <vector>
#include <string>
#include <complex>
#include <cstdint>

using namespace Rcpp;
using namespace mlx::core;
using mlx::core::complex64_t;

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
  if (dtype == "complex64") return complex64;
  if (dtype == "bool") return bool_;
  Rcpp::stop("Unsupported dtype: " + dtype);
}

// Helper: convert dtype to string
std::string dtype_to_string(Dtype dtype) {
  if (dtype == float32) return "float32";
  if (dtype == float64) return "float64";
  if (dtype == int32) return "int32";
  if (dtype == int64) return "int64";
  if (dtype == int16) return "int16";
  if (dtype == int8) return "int8";
  if (dtype == uint8) return "uint8";
  if (dtype == uint16) return "uint16";
  if (dtype == uint32) return "uint32";
  if (dtype == uint64) return "uint64";
  if (dtype == complex64) return "complex64";
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
SEXP cpp_mlx_from_r(SEXP x_, SEXP dim_, SEXP dtype_, SEXP device_) {
  IntegerVector dim(dim_);
  std::string dtype_str = as<std::string>(dtype_);
  std::string device_str = as<std::string>(device_);

  // Convert dimensions
  Shape shape(dim.begin(), dim.end());
  Dtype dt = string_to_dtype(dtype_str);
  StreamOrDevice dev = string_to_device(device_str);

  size_t ndim = shape.size();

  // R uses column-major, MLX uses row-major
  // Strategy: create array with reversed shape, then transpose
  Shape reversed_shape(shape.rbegin(), shape.rend());

  ComplexVector cx;
  NumericVector x;
  bool use_complex = (dt == complex64);
  if (use_complex) {
    cx = ComplexVector(x_);
  } else {
    x = NumericVector(x_);
  }

  // Create array directly from R data (column-major) using reversed shape
  // Use float32 for bool to avoid float64 GPU issues
  array arr_temp = [&]() -> array {
    if (dt == bool_) {
      std::vector<float> data_f32(x.begin(), x.end());
      return array(data_f32.data(), reversed_shape, float32);
    } else if (dt == float64) {
      return array(x.begin(), reversed_shape, float64);
    } else if (dt == float32) {
      std::vector<float> data_f32(x.begin(), x.end());
      return array(data_f32.data(), reversed_shape, float32);
    } else if (dt == complex64) {
      std::vector<complex64_t> data_c64(cx.size());
      for (R_xlen_t idx = 0; idx < cx.size(); ++idx) {
        const Rcomplex& val = cx[idx];
        data_c64[idx] = complex64_t(
            static_cast<float>(val.r),
            static_cast<float>(val.i));
      }
      return array(data_c64.data(), reversed_shape, complex64);
    } else {
      Rcpp::stop("Unsupported dtype for conversion from numeric");
    }
  }();

  // Transpose to correct orientation if multi-dimensional
  if (ndim > 1) {
    std::vector<int> perm(ndim);
    for (size_t i = 0; i < ndim; ++i) {
      perm[i] = static_cast<int>(ndim - 1 - i);
    }
    arr_temp = transpose(arr_temp, perm);
  }

  // Convert to target dtype and device
  array arr = astype(arr_temp, dt, dev);

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
SEXP cpp_mlx_to_r(SEXP xp_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  // Move to CPU if needed
  if (arr.dtype() == float64) {
    arr = astype(arr, float64, Device(Device::cpu));
  }

  size_t ndim = arr.ndim();

  // Transpose to column-major layout if multi-dimensional
  if (ndim > 1) {
    std::vector<int> perm(ndim);
    for (size_t i = 0; i < ndim; ++i) {
      perm[i] = static_cast<int>(ndim - 1 - i);
    }
    arr = transpose(arr, perm);
  }

  // Make contiguous and evaluate
  arr = contiguous(arr);
  eval(arr);

  // Get total size
  int total_size = arr.size();

  // Extract data based on dtype
  if (arr.dtype() == bool_) {
    LogicalVector result(total_size);
    const bool* data = arr.data<bool>();
    for (int i = 0; i < total_size; ++i) {
      result[i] = data[i];
    }
    return result;
  }

  if (arr.dtype() == int32) {
    IntegerVector result(total_size);
    const int32_t* data = arr.data<int32_t>();
    for (int i = 0; i < total_size; ++i) {
      result[i] = static_cast<int>(data[i]);
    }
    return result;
  }

  if (arr.dtype() == int64) {
    NumericVector result(total_size);
    const int64_t* data = arr.data<int64_t>();
    for (int i = 0; i < total_size; ++i) {
      result[i] = static_cast<double>(data[i]);
    }
    return result;
  }

  if (arr.dtype() == uint32) {
    NumericVector result(total_size);
    const uint32_t* data = arr.data<uint32_t>();
    for (int i = 0; i < total_size; ++i) {
      result[i] = static_cast<double>(data[i]);
    }
    return result;
  }

  if (arr.dtype() == uint64) {
    NumericVector result(total_size);
    const uint64_t* data = arr.data<uint64_t>();
    for (int i = 0; i < total_size; ++i) {
      result[i] = static_cast<double>(data[i]);
    }
    return result;
  }

  if (arr.dtype() == complex64) {
    ComplexVector result(total_size);
    const complex64_t* data = arr.data<complex64_t>();
    Rcomplex* out = reinterpret_cast<Rcomplex*>(result.begin());
    for (int i = 0; i < total_size; ++i) {
      out[i].r = static_cast<double>(data[i].real());
      out[i].i = static_cast<double>(data[i].imag());
    }
    return result;
  }

  // For float32/float64
  NumericVector result(total_size);
  if (arr.dtype() == float64) {
    const double* data = arr.data<double>();
    std::copy(data, data + total_size, result.begin());
  } else if (arr.dtype() == float32) {
    const float* data = arr.data<float>();
    for (int i = 0; i < total_size; ++i) {
      result[i] = static_cast<double>(data[i]);
    }
  } else {
    Rcpp::stop("Unsupported dtype for conversion to R");
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
