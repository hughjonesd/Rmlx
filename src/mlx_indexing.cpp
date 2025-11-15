// Indexing and slicing operations
#include "mlx_helpers.hpp"
#include <mlx/mlx.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

// [[Rcpp::export]]
SEXP cpp_mlx_where(SEXP cond_xp_, SEXP xp_true_, SEXP xp_false_,
                   std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* cond_wrapper = get_mlx_wrapper(cond_xp_);
  MlxArrayWrapper* true_wrapper = get_mlx_wrapper(xp_true_);
  MlxArrayWrapper* false_wrapper = get_mlx_wrapper(xp_false_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);

  array cond = astype(cond_wrapper->get(), bool_, target_device);
  array x = astype(true_wrapper->get(), target_dtype, target_device);
  array y = astype(false_wrapper->get(), target_dtype, target_device);

  array result = where(cond, x, y, target_device);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_take(SEXP xp_, SEXP indices_, int axis) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array idx_array = [&]() -> array {
    // Check if indices_ is an mlx array (external pointer)
    if (TYPEOF(indices_) == EXTPTRSXP) {
      MlxArrayWrapper* idx_wrapper = get_mlx_wrapper(indices_);
      array idx = idx_wrapper->get();
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
SEXP cpp_mlx_slice(SEXP xp_, SEXP starts_, SEXP stops_, SEXP strides_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  IntegerVector starts(starts_);
  IntegerVector stops(stops_);
  IntegerVector strides(strides_);

  // Convert to Shape
  Shape start_shape(starts.begin(), starts.end());
  Shape stop_shape(stops.begin(), stops.end());
  Shape stride_shape(strides.begin(), strides.end());

  array result = slice(wrapper->get(), start_shape, stop_shape, stride_shape);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_slice_update(SEXP xp_,
                          SEXP update_xp_,
                          IntegerVector start_,
                          IntegerVector stop_,
                          IntegerVector strides_) {
  MlxArrayWrapper* src_wrapper = get_mlx_wrapper(xp_);
  MlxArrayWrapper* update_wrapper = get_mlx_wrapper(update_xp_);

  Shape start_shape(start_.begin(), start_.end());
  Shape stop_shape(stop_.begin(), stop_.end());
  Shape stride_shape(strides_.begin(), strides_.end());

  array result = slice_update(src_wrapper->get(), update_wrapper->get(), start_shape, stop_shape, stride_shape);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_gather(SEXP xp_,
                    List indices_,
                    IntegerVector axes_,
                    std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  std::vector<array> indices;
  indices.reserve(indices_.size());
  for (int i = 0; i < indices_.size(); ++i) {
    List obj(indices_[i]);
    indices.push_back(get_mlx_wrapper(obj["ptr"])->get());
  }

  std::vector<int> axes(axes_.begin(), axes_.end());
  StreamOrDevice dev = string_to_device(device_str);
  Shape slice_sizes(axes.size(), 1);

  array result = gather(wrapper->get(), indices, axes, slice_sizes, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_scatter(SEXP xp_,
                     SEXP indices_xp_,
                     SEXP updates_xp_,
                     int axis) {
  MlxArrayWrapper* src_wrapper = get_mlx_wrapper(xp_);
  MlxArrayWrapper* idx_wrapper = get_mlx_wrapper(indices_xp_);
  MlxArrayWrapper* upd_wrapper = get_mlx_wrapper(updates_xp_);

  std::vector<array> indices_vec;
  indices_vec.push_back(idx_wrapper->get());
  std::vector<int> axes_vec{axis};

  array result = scatter(src_wrapper->get(), indices_vec, upd_wrapper->get(), axes_vec);
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
SEXP cpp_mlx_assign(SEXP xp_,
                    List normalized_,
                    SEXP updates_flat_xp_,
                    IntegerVector dim_sizes_) {
  MlxArrayWrapper* src_wrapper = get_mlx_wrapper(xp_);
  MlxArrayWrapper* updates_wrapper = get_mlx_wrapper(updates_flat_xp_);

  array src = src_wrapper->get();
  array updates = updates_wrapper->get();

  Shape dims(dim_sizes_.begin(), dim_sizes_.end());
  const int ndim = static_cast<int>(dims.size());
  if (normalized_.size() != ndim) {
    Rcpp::stop("Index count does not match array rank.");
  }

  std::vector<AxisSelection> axes(ndim);
  std::vector<int64_t> axis_lengths(ndim, 0);

  for (int axis = 0; axis < ndim; ++axis) {
    AxisSelection sel;
    SEXP idx = normalized_[axis];
    if (Rf_isNull(idx)) {
      sel.full = true;
      sel.length = dims[axis];
    } else {
      IntegerVector axis_idx(idx);
      sel.length = axis_idx.size();
      sel.values.assign(axis_idx.begin(), axis_idx.end());
    }
    axes[axis] = std::move(sel);
    axis_lengths[axis] = axes[axis].length;
  }

  size_t total = 1;
  for (int axis = 0; axis < ndim; ++axis) {
    total *= static_cast<size_t>(axis_lengths[axis]);
  }

  if (static_cast<size_t>(updates.size()) != total) {
    Rcpp::stop("Replacement value has incorrect length for selection.");
  }

  array flat_src = reshape(src, Shape{static_cast<int>(src.size())});
  array flat_updates = reshape(updates, Shape{static_cast<int>(total), 1});

  std::vector<int64_t> strides(ndim, 1);
  for (int axis = ndim - 2; axis >= 0; --axis) {
    strides[axis] = strides[axis + 1] * static_cast<int64_t>(dims[axis + 1]);
  }

  std::vector<int64_t> counters(ndim, 0);
  std::vector<int64_t> linear(total, 0);

  for (size_t idx = 0; idx < total; ++idx) {
    int64_t flat = 0;
    for (int axis = 0; axis < ndim; ++axis) {
      const AxisSelection& sel = axes[axis];
      int64_t coord = sel.full ? counters[axis] : sel.values[counters[axis]];
      flat += coord * strides[axis];
    }
    linear[idx] = flat;

    for (int axis = 0; axis < ndim; ++axis) {
      counters[axis]++;
      if (counters[axis] < axis_lengths[axis]) {
        break;
      }
      counters[axis] = 0;
    }
  }

  array idx_array(linear.data(), Shape{static_cast<int>(linear.size())}, int64);
  std::vector<array> idx_vec;
  idx_vec.push_back(idx_array);
  std::vector<int> axes_vec{0};

  array scattered = scatter(flat_src, idx_vec, flat_updates, axes_vec);
  array reshaped = reshape(scattered, src.shape());
  return make_mlx_xptr(std::move(reshaped));
}
