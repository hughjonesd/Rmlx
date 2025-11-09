#include "mlx_bindings.hpp"
#include <Rcpp.h>
#include <mlx/mlx.h>
#include <string>
#include <vector>
#include <complex>
#include <cstdint>

using namespace Rcpp;
using namespace mlx::core;
using mlx::core::complex64_t;

namespace rmlx {

namespace {

SEXP stream_tag() {
  static SEXP tag = Rf_install("Rmlx_stream");
  return tag;
}

} // namespace

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

void mlx_stream_finalizer(SEXP xp) {
  if (TYPEOF(xp) == EXTPTRSXP) {
    auto* wrapper = static_cast<MlxStreamWrapper*>(R_ExternalPtrAddr(xp));
    if (wrapper != nullptr) {
      delete wrapper;
      R_ClearExternalPtr(xp);
    }
  }
}

void mlx_imported_function_finalizer(SEXP xp) {
  if (TYPEOF(xp) == EXTPTRSXP) {
    auto* wrapper = static_cast<MlxImportedFunctionWrapper*>(R_ExternalPtrAddr(xp));
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

bool is_mlx_stream(SEXP value) {
  return TYPEOF(value) == EXTPTRSXP && R_ExternalPtrTag(value) == stream_tag() &&
         R_ExternalPtrAddr(value) != nullptr;
}

Stream get_mlx_stream(SEXP xp) {
  if (!is_mlx_stream(xp)) {
    Rcpp::stop("Expected an mlx_stream external pointer");
  }
  auto* wrapper = static_cast<MlxStreamWrapper*>(R_ExternalPtrAddr(xp));
  if (wrapper == nullptr) {
    Rcpp::stop("Invalid MLX stream pointer");
  }
  return wrapper->get();
}

MlxImportedFunctionWrapper* get_mlx_imported_function(SEXP xp) {
  if (TYPEOF(xp) != EXTPTRSXP) {
    Rcpp::stop("Expected external pointer");
  }
  auto* wrapper = static_cast<MlxImportedFunctionWrapper*>(R_ExternalPtrAddr(xp));
  if (wrapper == nullptr) {
    Rcpp::stop("Invalid MLX imported function pointer");
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

SEXP make_mlx_stream_xptr(Stream stream) {
  auto* wrapper = new MlxStreamWrapper(stream);
  SEXP xp = R_MakeExternalPtr(wrapper, stream_tag(), R_NilValue);
  R_RegisterCFinalizerEx(xp, mlx_stream_finalizer, TRUE);
  return xp;
}

SEXP make_mlx_imported_function_xptr(ImportedFunction function) {
  auto* wrapper = new MlxImportedFunctionWrapper(std::move(function));
  SEXP xp = R_MakeExternalPtr(wrapper, R_NilValue, R_NilValue);
  R_RegisterCFinalizerEx(xp, mlx_imported_function_finalizer, TRUE);
  return xp;
}

// Helper: convert dtype string to MLX dtype
Dtype string_to_dtype(const std::string& dtype) {
  if (dtype == "float32") return float32;
  if (dtype == "float64") return float64;
  if (dtype == "int8") return int8;
  if (dtype == "int16") return int16;
  if (dtype == "int32") return int32;
  if (dtype == "int64") return int64;
  if (dtype == "uint8") return uint8;
  if (dtype == "uint16") return uint16;
  if (dtype == "uint32") return uint32;
  if (dtype == "uint64") return uint64;
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

std::string device_to_string(const Device& device) {
  switch (device.type) {
  case Device::DeviceType::gpu:
    return "gpu";
  case Device::DeviceType::cpu:
    return "cpu";
  }
  Rcpp::stop("Unsupported device type");
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

  const bool col_major_fast_path =
    !use_complex &&
    TYPEOF(x_) == REALSXP &&
    ndim >= 1 &&
    (dt == float32 || dt == float64);

  if (col_major_fast_path) {
    size_t total = 1;
    for (size_t i = 0; i < ndim; ++i) {
      total *= static_cast<size_t>(shape[i]);
    }

    const Device original_device = default_device();
    const bool switch_to_cpu = original_device.type != Device::DeviceType::cpu;
    if (switch_to_cpu) {
      set_default_device(Device(Device::cpu));
    }

    try {
      array flat(
        REAL(x_),
        Shape{static_cast<int32_t>(total)},
        float64);

      Shape target_shape(shape.begin(), shape.end());
      Strides col_major_strides;
      col_major_strides.reserve(ndim);
      int64_t stride = 1;
      for (size_t i = 0; i < ndim; ++i) {
        col_major_strides.push_back(stride);
        stride *= static_cast<int64_t>(shape[i]);
      }

      array view = as_strided(
        flat,
        target_shape,
        col_major_strides,
        /*offset=*/0);

      array arr = contiguous(view, /*allow_col_major=*/true, Device(Device::cpu));
      if (arr.dtype() != dt) {
        arr = astype(arr, dt, Device(Device::cpu));
      }
      if (switch_to_cpu) {
        set_default_device(original_device);
      }
      if (device_str != "cpu") {
        arr = astype(arr, dt, dev);
      }
      return make_mlx_xptr(std::move(arr));
    } catch (...) {
      if (switch_to_cpu) {
        set_default_device(original_device);
      }
      throw;
    }
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
    } else if (dt == int8) {
      std::vector<int8_t> data_i8(x.begin(), x.end());
      return array(data_i8.data(), reversed_shape, int8);
    } else if (dt == int16) {
      std::vector<int16_t> data_i16(x.begin(), x.end());
      return array(data_i16.data(), reversed_shape, int16);
    } else if (dt == int32) {
      std::vector<int32_t> data_i32(x.begin(), x.end());
      return array(data_i32.data(), reversed_shape, int32);
    } else if (dt == int64) {
      std::vector<int64_t> data_i64(x.begin(), x.end());
      return array(data_i64.data(), reversed_shape, int64);
    } else if (dt == uint8) {
      std::vector<uint8_t> data_u8(x.begin(), x.end());
      return array(data_u8.data(), reversed_shape, uint8);
    } else if (dt == uint16) {
      std::vector<uint16_t> data_u16(x.begin(), x.end());
      return array(data_u16.data(), reversed_shape, uint16);
    } else if (dt == uint32) {
      std::vector<uint32_t> data_u32(x.begin(), x.end());
      return array(data_u32.data(), reversed_shape, uint32);
    } else if (dt == uint64) {
      std::vector<uint64_t> data_u64(x.begin(), x.end());
      return array(data_u64.data(), reversed_shape, uint64);
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

  // Integer types - int8, int16 -> R numeric (double)
  if (arr.dtype() == int8) {
    NumericVector result(total_size);
    const int8_t* data = arr.data<int8_t>();
    for (int i = 0; i < total_size; ++i) {
      result[i] = static_cast<double>(data[i]);
    }
    return result;
  }

  if (arr.dtype() == int16) {
    NumericVector result(total_size);
    const int16_t* data = arr.data<int16_t>();
    for (int i = 0; i < total_size; ++i) {
      result[i] = static_cast<double>(data[i]);
    }
    return result;
  }

  // int32 fits in R integer
  if (arr.dtype() == int32) {
    IntegerVector result(total_size);
    const int32_t* data = arr.data<int32_t>();
    for (int i = 0; i < total_size; ++i) {
      result[i] = static_cast<int>(data[i]);
    }
    return result;
  }

  // int64 -> R numeric (double)
  if (arr.dtype() == int64) {
    NumericVector result(total_size);
    const int64_t* data = arr.data<int64_t>();
    for (int i = 0; i < total_size; ++i) {
      result[i] = static_cast<double>(data[i]);
    }
    return result;
  }

  // Unsigned types - uint8, uint16 -> R numeric
  if (arr.dtype() == uint8) {
    NumericVector result(total_size);
    const uint8_t* data = arr.data<uint8_t>();
    for (int i = 0; i < total_size; ++i) {
      result[i] = static_cast<double>(data[i]);
    }
    return result;
  }

  if (arr.dtype() == uint16) {
    NumericVector result(total_size);
    const uint16_t* data = arr.data<uint16_t>();
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
