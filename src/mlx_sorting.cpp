// Sorting and partitioning operations
#include "mlx_helpers.hpp"
#include <mlx/mlx.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

// [[Rcpp::export]]
SEXP cpp_mlx_sort(SEXP xp_, Rcpp::Nullable<int> axis, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();
  StreamOrDevice dev = typed_device(arr.dtype(), device_str);

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int ax = Rcpp::as<int>(axis.get());
      ax = normalize_axis(arr, ax);
      return sort(arr, ax, dev);
    }
    return sort(arr, dev);
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_argsort(SEXP xp_, Rcpp::Nullable<int> axis, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();
  StreamOrDevice dev = typed_device(arr.dtype(), device_str);

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int ax = Rcpp::as<int>(axis.get());
      ax = normalize_axis(arr, ax);
      array idx = argsort(arr, ax, dev);
      idx = add(idx, array(1, idx.dtype()), dev);
      return idx;
    }
    array idx = argsort(arr, dev);
    idx = add(idx, array(1, idx.dtype()), dev);
    return idx;
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_topk(SEXP xp_, int k, Rcpp::Nullable<int> axis, std::string device_str) {
  if (k <= 0) {
    Rcpp::stop("k must be positive.");
  }

  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();
  StreamOrDevice dev = typed_device(arr.dtype(), device_str);

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int ax = Rcpp::as<int>(axis.get());
      ax = normalize_axis(arr, ax);
      int axis_size = static_cast<int>(arr.shape()[ax]);
      if (k > axis_size) {
        Rcpp::stop("k (%d) exceeds size of axis %d (%d).", k, ax + 1, axis_size);
      }
      return topk(arr, k, ax, dev);
    } else {
      int total = static_cast<int>(arr.size());
      if (k > total) {
        Rcpp::stop("k (%d) exceeds number of elements (%d).", k, total);
      }
      return topk(arr, k, dev);
    }
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_partition(SEXP xp_, int kth, Rcpp::Nullable<int> axis, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();
  StreamOrDevice dev = typed_device(arr.dtype(), device_str);

  if (kth < 0) {
    Rcpp::stop("kth must be non-negative.");
  }

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int target_axis = normalize_axis(arr, Rcpp::as<int>(axis.get()));
      int axis_size = static_cast<int>(arr.shape()[target_axis]);
      if (kth >= axis_size) {
        Rcpp::stop("kth (%d) exceeds size of axis %d (%d).", kth, target_axis + 1, axis_size);
      }
      return partition(arr, kth, target_axis, dev);
    }
    int total = static_cast<int>(arr.size());
    if (kth >= total) {
      Rcpp::stop("kth (%d) exceeds number of elements (%d).", kth, total);
    }
    return partition(arr, kth, dev);
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_argpartition(SEXP xp_,
                          int kth,
                          Rcpp::Nullable<int> axis,
                          std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();
  StreamOrDevice dev = typed_device(arr.dtype(), device_str);

  if (kth < 0) {
    Rcpp::stop("kth must be non-negative.");
  }

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int target_axis = normalize_axis(arr, Rcpp::as<int>(axis.get()));
      int axis_size = static_cast<int>(arr.shape()[target_axis]);
      if (kth >= axis_size) {
        Rcpp::stop("kth (%d) exceeds size of axis %d (%d).", kth, target_axis + 1, axis_size);
      }
      array idx = argpartition(arr, kth, target_axis, dev);
      idx = add(idx, array(1, idx.dtype()), dev);
      return idx;
    }
    int total = static_cast<int>(arr.size());
    if (kth >= total) {
      Rcpp::stop("kth (%d) exceeds number of elements (%d).", kth, total);
    }
    array idx = argpartition(arr, kth, dev);
    idx = add(idx, array(1, idx.dtype()), dev);
    return idx;
  }();
  return make_mlx_xptr(std::move(result));
}
