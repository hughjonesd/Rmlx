#ifndef MLX_HELPERS_HPP
#define MLX_HELPERS_HPP

#include "mlx_bindings.hpp"
#include <mlx/mlx.h>
#include <Rcpp.h>
#include <string>
#include <vector>
#include <optional>
#include <set>
#include <algorithm>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

namespace {

inline int normalize_axis(const array& arr, int axis) {
  int ndim = static_cast<int>(arr.ndim());
  if (axis < 0) {
    axis += ndim;
  }
  if (axis < 0 || axis >= ndim) {
    Rcpp::stop("Axis %d is out of bounds for array with %d dimensions.", axis + 1, ndim);
  }
  return axis;
}

inline std::vector<int> normalize_axes(const array& arr, const std::vector<int>& axes) {
  std::vector<int> result;
  result.reserve(axes.size());
  for (int axis : axes) {
    result.push_back(normalize_axis(arr, axis));
  }
  return result;
}

inline std::optional<std::vector<int>> optional_axes(
    const array& arr,
    const Rcpp::Nullable<Rcpp::IntegerVector>& axes) {
  if (axes.isNull()) {
    return std::nullopt;
  }
  Rcpp::IntegerVector axes_vec(axes.get());
  if (axes_vec.size() == 0) {
    Rcpp::stop("axes must contain at least one element.");
  }
  std::vector<int> raw_axes(axes_vec.begin(), axes_vec.end());
  for (int& axis : raw_axes) {
    if (axis == 0) {
      Rcpp::stop("Axis indices are 1-based; axis 0 is invalid.");
    }
    if (axis > 0) {
      axis -= 1;
    }
  }
  return normalize_axes(arr, raw_axes);
}

inline Dtype promote_numeric_dtype(Dtype lhs, Dtype rhs) {
  if (lhs == complex64 || rhs == complex64) {
    return complex64;
  }
  if (lhs == float64 || rhs == float64) {
    return float64;
  }
  if (lhs == float32 || rhs == float32) {
    return float32;
  }
  if (lhs == bool_ && rhs == bool_) {
    return bool_;
  }
  if ((lhs == bool_ && rhs == int32) || (rhs == bool_ && lhs == int32)) {
    return int32;
  }
  if ((lhs == bool_ && rhs == int64) || (rhs == bool_ && lhs == int64)) {
    return int64;
  }
  return float32;
}

inline std::vector<int> normalize_new_axes(const array& arr, const std::vector<int>& axes) {
  int ndim = static_cast<int>(arr.ndim());
  std::vector<int> normalized;
  normalized.reserve(axes.size());
  std::set<int> seen;
  for (int axis : axes) {
    int ax = axis;
    if (ax < 0) {
      ax += ndim + 1;
    }
    if (ax < 0 || ax > ndim) {
      Rcpp::stop("axis %d is out of bounds for array with %d dimensions.", axis, ndim);
    }
    if (!seen.insert(ax).second) {
      Rcpp::stop("axis %d is repeated.", axis);
    }
    normalized.push_back(ax);
  }
  std::sort(normalized.begin(), normalized.end());
  return normalized;
}

}  // namespace

#endif  // MLX_HELPERS_HPP
