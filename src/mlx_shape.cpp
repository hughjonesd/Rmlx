// Shape manipulation operations
#include "mlx_helpers.hpp"
#include <mlx/mlx.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

// [[Rcpp::export]]
SEXP cpp_mlx_transpose(SEXP xp_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array result = transpose(wrapper->get());
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_reshape(SEXP xp_, SEXP new_dim_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  IntegerVector new_dim(new_dim_);

  Shape shape(new_dim.begin(), new_dim.end());
  array result = reshape(wrapper->get(), shape);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_concat(SEXP args_, int axis) {
  List args(args_);
  if (args.size() == 0) {
    Rcpp::stop("No tensors supplied for concatenation.");
  }
  std::vector<array> arrays;
  arrays.reserve(args.size());
  std::string device_str;
  for (int i = 0; i < args.size(); ++i) {
    List obj(args[i]);
    arrays.push_back(get_mlx_wrapper(obj["ptr"])->get());
    if (i == 0) {
      device_str = Rcpp::as<std::string>(obj["device"]);
    }
  }

  StreamOrDevice dev = string_to_device(device_str);
  array result = concatenate(arrays, axis, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_stack(SEXP args_, int axis, std::string device_str) {
  List args(args_);
  if (args.size() == 0) {
    Rcpp::stop("No tensors supplied for stacking.");
  }

  std::vector<array> arrays;
  arrays.reserve(args.size());
  for (int i = 0; i < args.size(); ++i) {
    List obj(args[i]);
    arrays.push_back(get_mlx_wrapper(obj["ptr"])->get());
  }

  StreamOrDevice dev = string_to_device(device_str);
  array result = stack(arrays, axis, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_squeeze(SEXP xp_, Rcpp::Nullable<Rcpp::IntegerVector> axes) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = [&]() -> array {
    if (axes.isNotNull()) {
      Rcpp::IntegerVector axes_vec(axes.get());
      std::vector<int> ax(axes_vec.begin(), axes_vec.end());
      return squeeze(arr, normalize_axes(arr, ax));
    }
    return squeeze(arr);
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_expand_dims(SEXP xp_, Rcpp::IntegerVector axes_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  std::vector<int> axes(axes_.begin(), axes_.end());
  std::vector<int> normalized = normalize_new_axes(arr, axes);
  array result = expand_dims(arr, normalized);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_repeat(SEXP xp_, int repeats, Rcpp::Nullable<int> axis) {
  if (repeats <= 0) {
    Rcpp::stop("repeats must be positive.");
  }
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = [&]() -> array {
    if (axis.isNotNull()) {
      int ax = normalize_axis(arr, Rcpp::as<int>(axis.get()));
      return repeat(arr, repeats, ax);
    }
    return repeat(arr, repeats);
  }();
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_tile(SEXP xp_, Rcpp::IntegerVector reps_) {
  if (reps_.size() == 0) {
    Rcpp::stop("reps must contain at least one element.");
  }
  std::vector<int> reps(reps_.begin(), reps_.end());
  for (int value : reps) {
    if (value <= 0) {
      Rcpp::stop("All repetitions must be positive.");
    }
  }

  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = tile(arr, reps);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_roll(SEXP xp_, SEXP shift_, Rcpp::Nullable<Rcpp::IntegerVector> axes_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  NumericVector shifts(shift_);
  if (shifts.size() == 0) {
    Rcpp::stop("shift must contain at least one element.");
  }

  array result = [&]() -> array {
    if (axes_.isNotNull()) {
      Rcpp::IntegerVector axes_vec(axes_.get());
      if (axes_vec.size() != shifts.size()) {
        Rcpp::stop("shift and axis must have the same length.");
      }
      std::vector<int> axes(axes_vec.begin(), axes_vec.end());
      axes = normalize_axes(arr, axes);
      Shape shift_shape;
      shift_shape.reserve(shifts.size());
      for (double val : shifts) {
        shift_shape.push_back(static_cast<int>(val));
      }
      if (axes.size() == 1) {
        return roll(arr, static_cast<int>(shifts[0]), axes[0]);
      }
      return roll(arr, shift_shape, axes);
    }

    if (shifts.size() == 1) {
      return roll(arr, static_cast<int>(shifts[0]));
    }
    Shape shift_shape;
    shift_shape.reserve(shifts.size());
    for (double val : shifts) {
      shift_shape.push_back(static_cast<int>(val));
    }
    return roll(arr, shift_shape);
  }();

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_moveaxis(SEXP xp_, Rcpp::IntegerVector source_, Rcpp::IntegerVector destination_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  int ndim = static_cast<int>(arr.ndim());
  int n = source_.size();
  if (n == 0) {
    Rcpp::stop("source must contain at least one axis.");
  }
  if (destination_.size() != n) {
    Rcpp::stop("source and destination must have the same length.");
  }

  std::vector<int> source_norm;
  std::vector<int> dest_norm;
  source_norm.reserve(n);
  dest_norm.reserve(n);

  std::vector<bool> is_moved(ndim, false);
  for (int axis : source_) {
    int norm = normalize_axis(arr, axis);
    if (is_moved[norm]) {
      Rcpp::stop("source axes must be unique.");
    }
    is_moved[norm] = true;
    source_norm.push_back(norm);
  }

  std::vector<bool> dest_seen(ndim, false);
  for (int axis : destination_) {
    int norm = normalize_axis(arr, axis);
    if (dest_seen[norm]) {
      Rcpp::stop("destination axes must be unique.");
    }
    dest_seen[norm] = true;
    dest_norm.push_back(norm);
  }

  std::vector<std::pair<int, int>> moves;
  moves.reserve(n);
  for (int i = 0; i < n; ++i) {
    moves.emplace_back(dest_norm[i], source_norm[i]);
  }
  std::sort(
      moves.begin(),
      moves.end(),
      [](const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) {
        return lhs.first < rhs.first;
      });

  std::vector<int> remaining;
  remaining.reserve(ndim - n);
  for (int axis = 0; axis < ndim; ++axis) {
    if (!is_moved[axis]) {
      remaining.push_back(axis);
    }
  }

  std::vector<int> permutation;
  permutation.reserve(ndim);
  std::size_t move_idx = 0;
  std::size_t rem_idx = 0;
  for (int pos = 0; pos < ndim; ++pos) {
    if (move_idx < moves.size() && moves[move_idx].first == pos) {
      permutation.push_back(moves[move_idx].second);
      ++move_idx;
    } else {
      if (rem_idx >= remaining.size()) {
        Rcpp::stop("Invalid moveaxis configuration.");
      }
      permutation.push_back(remaining[rem_idx]);
      ++rem_idx;
    }
  }

  array result = transpose(arr, permutation);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_flatten(SEXP xp_, int start_axis, int end_axis) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = flatten(arr, start_axis, end_axis);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_swapaxes(SEXP xp_, int axis1, int axis2) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  array result = swapaxes(arr, axis1, axis2);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_meshgrid(SEXP args_, bool sparse, std::string indexing, std::string device_str) {
  List args(args_);
  if (args.size() == 0) {
    Rcpp::stop("No tensors supplied for meshgrid.");
  }

  std::vector<array> arrays;
  arrays.reserve(args.size());
  for (int i = 0; i < args.size(); ++i) {
    List obj(args[i]);
    arrays.push_back(get_mlx_wrapper(obj["ptr"])->get());
  }

  StreamOrDevice dev = string_to_device(device_str);
  std::vector<array> result = meshgrid(arrays, sparse, indexing, dev);

  List out(result.size());
  for (std::size_t i = 0; i < result.size(); ++i) {
    out[i] = make_mlx_xptr(std::move(result[i]));
  }
  return out;
}

// [[Rcpp::export]]
SEXP cpp_mlx_broadcast_to(SEXP xp_, Rcpp::IntegerVector shape_, std::string device_str) {
  if (shape_.size() == 0) {
    Rcpp::stop("shape must contain at least one element.");
  }

  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  Shape shape(shape_.begin(), shape_.end());
  StreamOrDevice dev = string_to_device(device_str);

  array result = broadcast_to(arr, shape, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_broadcast_arrays(SEXP args_, std::string device_str) {
  List args(args_);
  if (args.size() == 0) {
    Rcpp::stop("No tensors supplied for broadcast.");
  }

  std::vector<array> arrays;
  arrays.reserve(args.size());
  for (int i = 0; i < args.size(); ++i) {
    List obj(args[i]);
    arrays.push_back(get_mlx_wrapper(obj["ptr"])->get());
  }

  StreamOrDevice dev = string_to_device(device_str);
  std::vector<array> result = broadcast_arrays(arrays, dev);

  List out(result.size());
  for (std::size_t i = 0; i < result.size(); ++i) {
    out[i] = make_mlx_xptr(std::move(result[i]));
  }
  return out;
}

// [[Rcpp::export]]
SEXP cpp_mlx_pad(SEXP xp_,
                 Rcpp::IntegerMatrix pad_pairs_,
                 double pad_value,
                 std::string dtype_str,
                 std::string device_str,
                 std::string mode_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  if (pad_pairs_.ncol() != 2) {
    Rcpp::stop("pad_width must have two columns (before, after).");
  }
  int ndim = static_cast<int>(arr.ndim());
  if (pad_pairs_.nrow() != ndim) {
    Rcpp::stop("pad_width row count must match tensor rank.");
  }

  std::vector<std::pair<int, int>> pad_width;
  pad_width.reserve(ndim);
  for (int i = 0; i < pad_pairs_.nrow(); ++i) {
    int before = pad_pairs_(i, 0);
    int after = pad_pairs_(i, 1);
    if (before < 0 || after < 0) {
      Rcpp::stop("pad widths must be non-negative.");
    }
    pad_width.emplace_back(before, after);
  }

  Dtype dtype = string_to_dtype(dtype_str);
  StreamOrDevice dev = string_to_device(device_str);
  array pad_val = array(pad_value, dtype);

  array result = pad(arr, pad_width, pad_val, mode_str, dev);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_split(SEXP xp_,
                   Rcpp::Nullable<int> num_splits_,
                   Rcpp::Nullable<Rcpp::IntegerVector> indices_,
                   int axis,
                   std::string dtype_str,
                   std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  int ax = normalize_axis(arr, axis);
  StreamOrDevice dev = string_to_device(device_str);

  std::vector<array> outputs;

  if (indices_.isNotNull()) {
    Rcpp::IntegerVector indices_vec(indices_.get());
    if (indices_vec.size() == 0) {
      Rcpp::stop("indices must contain at least one value.");
    }
    Shape indices_shape;
    indices_shape.reserve(indices_vec.size());
    for (int value : indices_vec) {
      if (value < 0) {
        Rcpp::stop("Split indices must be non-negative.");
      }
      indices_shape.push_back(value);
    }
    outputs = split(arr, indices_shape, ax, dev);
  } else if (num_splits_.isNotNull()) {
    int num_splits = Rcpp::as<int>(num_splits_.get());
    if (num_splits <= 0) {
      Rcpp::stop("num_splits must be positive.");
    }
    outputs = split(arr, num_splits, ax, dev);
  } else {
    Rcpp::stop("Either num_splits or indices must be supplied.");
  }

  List out(outputs.size());
  for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
    out[i] = make_mlx_xptr(std::move(outputs[i]));
  }
  return out;
}

// [[Rcpp::export]]
SEXP cpp_mlx_unflatten(SEXP a_xp_, int axis, IntegerVector shape, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array a_cpu = astype(a_wrapper->get(), a_wrapper->get().dtype(), cpu_stream);

  // Convert 1-indexed to 0-indexed
  int ax = axis - 1;
  ax = normalize_axis(a_cpu, ax);

  // Convert shape to SmallVector
  Shape new_shape;
  for (int i = 0; i < shape.size(); i++) {
    new_shape.push_back(shape[i]);
  }

  array result_cpu = unflatten(a_cpu, ax, new_shape, cpu_stream);
  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_contiguous(SEXP xp_, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice target_device = string_to_device(device_str);
  array on_device = astype(arr, arr.dtype(), target_device);
  array result = contiguous(on_device);
  return make_mlx_xptr(std::move(result));
}
