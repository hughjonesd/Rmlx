#include "mlx_bindings.hpp"
#include "colmajor_helpers.hpp"
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

// Finalizer for R external pointer
void mlx_array_finalizer(SEXP xp) {
  if (TYPEOF(xp) == EXTPTRSXP) {
    auto* arr = static_cast<array*>(R_ExternalPtrAddr(xp));
    if (arr != nullptr) {
      delete arr;
      R_ClearExternalPtr(xp);
    }
  }
}

void mlx_stream_finalizer(SEXP xp) {
  if (TYPEOF(xp) == EXTPTRSXP) {
    auto* stream = static_cast<Stream*>(R_ExternalPtrAddr(xp));
    if (stream != nullptr) {
      delete stream;
      R_ClearExternalPtr(xp);
    }
  }
}

void mlx_imported_function_finalizer(SEXP xp) {
  if (TYPEOF(xp) == EXTPTRSXP) {
    auto* function = static_cast<ImportedFunction*>(R_ExternalPtrAddr(xp));
    if (function != nullptr) {
      delete function;
      R_ClearExternalPtr(xp);
    }
  }
}

// Get the native array from its external pointer
array& get_mlx_array(SEXP xp) {
  if (TYPEOF(xp) != EXTPTRSXP) {
    Rcpp::stop("Expected external pointer");
  }
  auto* arr = static_cast<array*>(R_ExternalPtrAddr(xp));
  if (arr == nullptr) {
    Rcpp::stop("Invalid MLX array pointer");
  }
  return *arr;
}

bool is_mlx_stream(SEXP value) {
  return TYPEOF(value) == EXTPTRSXP && R_ExternalPtrTag(value) == stream_tag() &&
         R_ExternalPtrAddr(value) != nullptr;
}

Stream get_mlx_stream(SEXP xp) {
  if (!is_mlx_stream(xp)) {
    Rcpp::stop("Expected an mlx_stream external pointer");
  }
  auto* stream = static_cast<Stream*>(R_ExternalPtrAddr(xp));
  if (stream == nullptr) {
    Rcpp::stop("Invalid MLX stream pointer");
  }
  return *stream;
}

const ImportedFunction& get_mlx_imported_function(SEXP xp) {
  if (TYPEOF(xp) != EXTPTRSXP) {
    Rcpp::stop("Expected external pointer");
  }
  auto* function = static_cast<ImportedFunction*>(R_ExternalPtrAddr(xp));
  if (function == nullptr) {
    Rcpp::stop("Invalid MLX imported function pointer");
  }
  return *function;
}

// Wrap MLX array in external pointer
SEXP make_mlx_xptr(const array& arr) {
  auto* ptr = new array(arr);
  SEXP xp = R_MakeExternalPtr(ptr, R_NilValue, R_NilValue);
  R_RegisterCFinalizerEx(xp, mlx_array_finalizer, TRUE);
  return xp;
}

SEXP make_mlx_xptr(array&& arr) {
  auto* ptr = new array(std::move(arr));
  SEXP xp = R_MakeExternalPtr(ptr, R_NilValue, R_NilValue);
  R_RegisterCFinalizerEx(xp, mlx_array_finalizer, TRUE);
  return xp;
}

SEXP make_mlx_stream_xptr(Stream stream) {
  auto* ptr = new Stream(stream);
  SEXP xp = R_MakeExternalPtr(ptr, stream_tag(), R_NilValue);
  R_RegisterCFinalizerEx(xp, mlx_stream_finalizer, TRUE);
  return xp;
}

SEXP make_mlx_imported_function_xptr(ImportedFunction function) {
  auto* ptr = new ImportedFunction(std::move(function));
  SEXP xp = R_MakeExternalPtr(ptr, R_NilValue, R_NilValue);
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
SEXP cpp_mlx_from_r(SEXP x_, SEXP dim_, SEXP dtype_) {
  IntegerVector dim(dim_);
  std::string dtype_str = as<std::string>(dtype_);

  // Convert dimensions
  Shape shape(dim.begin(), dim.end());
  Dtype dt = string_to_dtype(dtype_str);

  size_t ndim = shape.size();

  // R uses column-major, MLX uses row-major
  // Strategy: create array with reversed shape, then transpose
  Shape reversed_shape(shape.rbegin(), shape.rend());

  // The iterator constructor copies and casts directly into MLX-owned storage.
  array arr = [&]() -> array {
    if (dt == complex64) {
      ComplexVector cx(x_);
      std::vector<complex64_t> data_c64(cx.size());
      for (R_xlen_t idx = 0; idx < cx.size(); ++idx) {
        const Rcomplex& val = cx[idx];
        data_c64[idx] = complex64_t(
            static_cast<float>(val.r),
            static_cast<float>(val.i));
      }
      return array(data_c64.data(), reversed_shape, complex64);
    }

    // Dispatch before constructing NumericVector, which would coerce to double.
    switch (TYPEOF(x_)) {
    case INTSXP:
      return array(INTEGER(x_), reversed_shape, dt);
    case LGLSXP:
      return array(LOGICAL(x_), reversed_shape, dt);
    default:
      NumericVector x(x_);
      return array(x.begin(), reversed_shape, dt);
    }
  }();

  // Transpose to correct orientation if multi-dimensional.
  if (ndim > 1) {
    arr = transpose_between_mlx_and_r(arr);
  }

  return make_mlx_xptr(std::move(arr));
}

// [[Rcpp::export]]
SEXP cpp_mlx_to_r(SEXP xp_) {
  array arr = get_mlx_array(xp_);

  size_t ndim = arr.ndim();

  // Transpose to column-major layout if multi-dimensional
  if (ndim > 1) {
    arr = transpose_between_mlx_and_r(arr);
  }

  arr = contiguous(arr, /*allow_col_major=*/false);
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
  array& arr_ref = get_mlx_array(xp_);
  eval(arr_ref);
}

// [[Rcpp::export]]
IntegerVector cpp_mlx_shape(SEXP xp_) {
  array& arr_ref = get_mlx_array(xp_);
  const auto& shape = arr_ref.shape();

  IntegerVector result(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    result[i] = shape[i];
  }

  return result;
}

// [[Rcpp::export]]
std::string cpp_mlx_dtype(SEXP xp_) {
  array& arr_ref = get_mlx_array(xp_);
  return dtype_to_string(arr_ref.dtype());
}
