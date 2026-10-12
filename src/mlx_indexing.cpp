// Indexing and slicing operations
#include "mlx_helpers.hpp"
#include "colmajor_helpers.hpp"
#include <algorithm>
#include <mlx/mlx.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

// [[Rcpp::export]]
SEXP cpp_mlx_where(SEXP cond_xp_, SEXP xp_true_, SEXP xp_false_,
                   std::string dtype_str) {
  array& cond_ref = get_mlx_array(cond_xp_);
  array& true_ref = get_mlx_array(xp_true_);
  array& false_ref = get_mlx_array(xp_false_);

  Dtype target_dtype = string_to_dtype(dtype_str);

  array cond = astype(cond_ref, bool_);
  array x = astype(true_ref, target_dtype);
  array y = astype(false_ref, target_dtype);

  array result = where(cond, x, y);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_take(SEXP xp_, SEXP indices_, int axis) {
  array arr = get_mlx_array(xp_);

  array idx_array = [&]() -> array {
    // Check if indices_ is an mlx array (external pointer)
    if (TYPEOF(indices_) == EXTPTRSXP) {
      array idx = get_mlx_array(indices_);
      // Ensure index array is at least 1D to match R vector behavior
      // If scalar, reshape to [1] so take() preserves dimensions correctly
      if (idx.ndim() == 0) {
        idx = reshape(idx, {1});
      }
      return idx;
    } else {
      // Handle R integer vector
      IntegerVector idx(indices_);
      std::vector<int64_t> data(idx.begin(), idx.end());
      Shape shape{static_cast<int>(data.size())};
      return array(data.data(), shape, int64);
    }
  }();

  array result = take(arr, idx_array, axis);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_take_along_axis(SEXP xp_, SEXP indices_xp_, int axis) {
  array& arr_ref = get_mlx_array(xp_);
  array& idx_ref = get_mlx_array(indices_xp_);
  array result = take_along_axis(
    arr_ref,
    idx_ref,
    axis);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_put_along_axis(SEXP xp_, SEXP indices_xp_, SEXP values_xp_, int axis) {
  array& arr_ref = get_mlx_array(xp_);
  array& idx_ref = get_mlx_array(indices_xp_);
  array& values_ref = get_mlx_array(values_xp_);
  array result = put_along_axis(
    arr_ref,
    idx_ref,
    values_ref,
    axis
  );
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_scatter_add_axis(SEXP xp_, SEXP indices_xp_, SEXP values_xp_, int axis) {
  array& arr_ref = get_mlx_array(xp_);
  array& idx_ref = get_mlx_array(indices_xp_);
  array& values_ref = get_mlx_array(values_xp_);
  array result = scatter_add_axis(
    arr_ref,
    idx_ref,
    values_ref,
    axis
  );
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_slice(SEXP xp_, SEXP starts_, SEXP stops_, SEXP strides_) {
  array& arr_ref = get_mlx_array(xp_);
  IntegerVector starts(starts_);
  IntegerVector stops(stops_);
  IntegerVector strides(strides_);

  // Convert to Shape
  Shape start_shape(starts.begin(), starts.end());
  Shape stop_shape(stops.begin(), stops.end());
  Shape stride_shape(strides.begin(), strides.end());

  array result = slice(arr_ref, start_shape, stop_shape, stride_shape);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_slice_update(SEXP xp_,
                          SEXP update_xp_,
                          IntegerVector start_,
                          IntegerVector stop_,
                          IntegerVector strides_) {
  array& src_ref = get_mlx_array(xp_);
  array& update_ref = get_mlx_array(update_xp_);

  Shape start_shape(start_.begin(), start_.end());
  Shape stop_shape(stop_.begin(), stop_.end());
  Shape stride_shape(strides_.begin(), strides_.end());

  array result = slice_update(
    src_ref,
    update_ref,
    start_shape,
    stop_shape,
    stride_shape);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_gather(SEXP xp_,
                    List indices_,
                    IntegerVector axes_) {
  array src = get_mlx_array(xp_);
  std::vector<array> indices;
  indices.reserve(indices_.size());
  for (int i = 0; i < indices_.size(); ++i) {
    List obj(indices_[i]);
    indices.push_back(get_mlx_array(obj["ptr"]));
  }

  std::vector<int> axes(axes_.begin(), axes_.end());
  Shape slice_sizes(src.ndim(), 0);
  Shape src_shape = src.shape();
  for (int i = 0; i < src.ndim(); ++i) {
    const bool is_gather_axis = std::find(axes.begin(), axes.end(), i) != axes.end();
    slice_sizes[i] = is_gather_axis ? 1 : src_shape[i];
  }

  array result = gather(src, indices, axes, slice_sizes);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_scatter(SEXP xp_,
                     List indices_,
                     SEXP updates_xp_,
                     IntegerVector axes_) {
  array src = get_mlx_array(xp_);
  array& upd_ref = get_mlx_array(updates_xp_);

  std::vector<array> indices_vec;
  indices_vec.reserve(indices_.size());
  for (int i = 0; i < indices_.size(); ++i) {
    List obj(indices_[i]);
    indices_vec.push_back(get_mlx_array(obj["ptr"]));
  }

  std::vector<int> axes_vec(axes_.begin(), axes_.end());

  array result = scatter(src, indices_vec, upd_ref, axes_vec);
  return make_mlx_xptr(std::move(result));
}

namespace {

struct AxisSelection {
  bool full = false;
  std::vector<int64_t> values;
  int64_t length = 0;
};

} // namespace

// [[Rcpp::export]]
SEXP cpp_mlx_masked_scatter(SEXP xp_,
                                     SEXP mask_xp_,
                                     SEXP updates_xp_) {
  array& src_ref = get_mlx_array(xp_);
  array& mask_ref = get_mlx_array(mask_xp_);
  array& updates_ref = get_mlx_array(updates_xp_);

  array src = transpose_between_mlx_and_r(src_ref);
  array mask = transpose_between_mlx_and_r(astype(mask_ref, bool_));
  array updates = astype(updates_ref, src.dtype());

  array result = masked_scatter(src, mask, updates);
  result = transpose_between_mlx_and_r(result);
  return make_mlx_xptr(std::move(result));
}
