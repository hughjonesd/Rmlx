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

// Helper: Reorder data from R (column-major) to MLX (row-major) for 2D arrays
// Instead of transpose, we physically rearrange elements
std::vector<double> reorder_col_major_to_row_major(const double* data, int nrow, int ncol) {
  std::vector<double> result(nrow * ncol);
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      result[i * ncol + j] = data[j * nrow + i];  // col-major[j,i] -> row-major[i,j]
    }
  }
  return result;
}

std::vector<float> reorder_col_major_to_row_major_f32(const double* data, int nrow, int ncol) {
  std::vector<float> result(nrow * ncol);
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      result[i * ncol + j] = static_cast<float>(data[j * nrow + i]);
    }
  }
  return result;
}

template <typename SrcT, typename DstT>
void reorder_col_major_to_row_major_nd(const SrcT* src,
                                       DstT* dst,
                                       const std::vector<int>& dims) {
  size_t ndim = dims.size();
  size_t total = 1;
  for (int dim : dims) {
    total *= static_cast<size_t>(dim);
  }

  if (ndim <= 1) {
    for (size_t idx = 0; idx < total; ++idx) {
      dst[idx] = static_cast<DstT>(src[idx]);
    }
    return;
  }

  std::vector<size_t> row_strides(ndim);
  row_strides[ndim - 1] = 1;
  for (int axis = static_cast<int>(ndim) - 2; axis >= 0; --axis) {
    row_strides[axis] = row_strides[axis + 1] * static_cast<size_t>(dims[axis + 1]);
  }

  for (size_t index = 0; index < total; ++index) {
    size_t remainder = index;
    size_t row_index = 0;

    for (size_t axis = 0; axis < ndim; ++axis) {
      size_t coord = remainder % static_cast<size_t>(dims[axis]);
      remainder /= static_cast<size_t>(dims[axis]);
      row_index += coord * row_strides[axis];
    }

    dst[row_index] = static_cast<DstT>(src[index]);
  }
}

// Helper: Reorder data from MLX (row-major) to R (column-major) for 2D arrays
void reorder_row_major_to_col_major(const double* src, double* dst, int nrow, int ncol) {
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      dst[j * nrow + i] = src[i * ncol + j];  // row-major[i,j] -> col-major[j,i]
    }
  }
}

void reorder_row_major_to_col_major_f32(const float* src, double* dst, int nrow, int ncol) {
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      dst[j * nrow + i] = static_cast<double>(src[i * ncol + j]);
    }
  }
}

template <typename SrcT, typename DstT>
void reorder_row_major_to_col_major_nd(const SrcT* src,
                                       DstT* dst,
                                       const std::vector<int>& dims) {
  size_t ndim = dims.size();
  size_t total = 1;
  for (int dim : dims) {
    total *= static_cast<size_t>(dim);
  }

  if (ndim <= 1) {
    for (size_t idx = 0; idx < total; ++idx) {
      dst[idx] = static_cast<DstT>(src[idx]);
    }
    return;
  }

  std::vector<size_t> col_strides(ndim);
  col_strides[0] = 1;
  for (size_t axis = 1; axis < ndim; ++axis) {
    col_strides[axis] = col_strides[axis - 1] * static_cast<size_t>(dims[axis - 1]);
  }

  std::vector<size_t> row_strides(ndim);
  row_strides[ndim - 1] = 1;
  for (int axis = static_cast<int>(ndim) - 2; axis >= 0; --axis) {
    row_strides[axis] = row_strides[axis + 1] * static_cast<size_t>(dims[axis + 1]);
  }

  for (size_t index = 0; index < total; ++index) {
    size_t remainder = index;
    size_t col_index = 0;

    for (size_t axis = 0; axis < ndim; ++axis) {
      size_t coord;
      if (axis == ndim - 1) {
        coord = remainder;
      } else {
        coord = remainder / row_strides[axis];
        remainder %= row_strides[axis];
      }
      col_index += coord * col_strides[axis];
    }

    dst[col_index] = static_cast<DstT>(src[index]);
  }
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

  size_t ndim = shape.size();
  std::vector<int> dims(shape.begin(), shape.end());

  auto make_array = [&](Dtype storage_dtype) -> array {
    if (ndim == 2) {
      if (storage_dtype == float64) {
        std::vector<double> data_reordered = reorder_col_major_to_row_major(x.begin(), shape[0], shape[1]);
        return array(data_reordered.data(), shape, float64);
      } else if (storage_dtype == float32) {
        std::vector<float> data_reordered = reorder_col_major_to_row_major_f32(x.begin(), shape[0], shape[1]);
        return array(data_reordered.data(), shape, float32);
      } else {
        Rcpp::stop("Unsupported dtype for conversion from numeric");
      }
    } else if (ndim <= 1) {
      if (storage_dtype == float64) {
        return array(x.begin(), shape, float64);
      } else if (storage_dtype == float32) {
        std::vector<float> data_f32(x.begin(), x.end());
        return array(data_f32.begin(), shape, float32);
      } else {
        Rcpp::stop("Unsupported dtype for conversion from numeric");
      }
    } else {
      size_t total = x.size();
      if (storage_dtype == float64) {
        std::vector<double> data_reordered(total);
        reorder_col_major_to_row_major_nd<double, double>(x.begin(), data_reordered.data(), dims);
        return array(data_reordered.data(), shape, float64);
      } else if (storage_dtype == float32) {
        std::vector<float> data_reordered(total);
        reorder_col_major_to_row_major_nd<double, float>(x.begin(), data_reordered.data(), dims);
        return array(data_reordered.data(), shape, float32);
      } else {
        Rcpp::stop("Unsupported dtype for conversion from numeric");
      }
    }
  };

  // Copy data from R to MLX, reordering for layout if needed
  array arr_cpu = [&]() -> array {
    if (dt == float64) {
      return make_array(float64);
    } else if (dt == float32) {
      return make_array(float32);
    } else if (dt == bool_) {
      array float_arr = make_array(float32);
      return astype(float_arr, bool_);
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
SEXP cpp_mlx_to_r(SEXP xp_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  // Ensure array is row-contiguous before extracting data
  // This is important for transposed arrays which may have non-standard strides
  if (arr.ndim() >= 2) {
    arr = contiguous(arr);
  }

  // Evaluate first
  eval(arr);

  // Get total size and shape
  int total_size = arr.size();
  const auto& shape = arr.shape();

  // For boolean arrays, return LogicalVector
  if (arr.dtype() == bool_) {
    LogicalVector result(total_size);
    const bool* data = arr.data<bool>();
    std::vector<int> dims(shape.begin(), shape.end());

    if (arr.ndim() == 2) {
      int nrow = shape[0];
      int ncol = shape[1];
      for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
          result[j * nrow + i] = data[i * ncol + j];
        }
      }
    } else if (arr.ndim() <= 1) {
      for (int i = 0; i < total_size; ++i) {
        result[i] = data[i];
      }
    } else {
      reorder_row_major_to_col_major_nd<bool, int>(data, result.begin(), dims);
    }
    return result;
  }

  // For numeric arrays, return NumericVector
  NumericVector result(total_size);
  std::vector<int> dims(shape.begin(), shape.end());

  // Copy data, reordering for layout if needed
  if (arr.ndim() == 2) {
    // 2D: reorder from row-major to column-major
    int nrow = shape[0];
    int ncol = shape[1];
    if (arr.dtype() == float64) {
      const double* data = arr.data<double>();
      reorder_row_major_to_col_major(data, result.begin(), nrow, ncol);
    } else if (arr.dtype() == float32) {
      const float* data = arr.data<float>();
      reorder_row_major_to_col_major_f32(data, result.begin(), nrow, ncol);
    } else {
      Rcpp::stop("Unsupported dtype for conversion to R");
    }
  } else if (arr.ndim() <= 1) {
    // 1D: no reordering needed
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
  } else {
    if (arr.dtype() == float64) {
      const double* data = arr.data<double>();
      reorder_row_major_to_col_major_nd<double, double>(data, result.begin(), dims);
    } else if (arr.dtype() == float32) {
      const float* data = arr.data<float>();
      reorder_row_major_to_col_major_nd<float, double>(data, result.begin(), dims);
    } else {
      Rcpp::stop("Unsupported dtype for conversion to R");
    }
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
