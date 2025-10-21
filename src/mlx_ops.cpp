#include "mlx_bindings.hpp"
#include <mlx/mlx.h>
#include <mlx/linalg.h>
#include <mlx/fft.h>
#include <mlx/random.h>
#include <Rcpp.h>
#include <string>
#include <numeric>
#include <algorithm>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

// Unary operations
// [[Rcpp::export]]
SEXP cpp_mlx_unary(SEXP xp_, std::string op) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  array result = [&]() -> array {
    if (op == "neg") {
      return negative(wrapper->get());
    } else if (op == "abs") {
      return abs(wrapper->get());
    } else if (op == "sign") {
      return sign(wrapper->get());
    } else if (op == "sqrt") {
      return sqrt(wrapper->get());
    } else if (op == "rsqrt") {
      return rsqrt(wrapper->get());
    } else if (op == "square") {
      return square(wrapper->get());
    } else if (op == "exp") {
      return exp(wrapper->get());
    } else if (op == "expm1") {
      return expm1(wrapper->get());
    } else if (op == "log") {
      return log(wrapper->get());
    } else if (op == "log2") {
      return log2(wrapper->get());
    } else if (op == "log10") {
      return log10(wrapper->get());
    } else if (op == "log1p") {
      return log1p(wrapper->get());
    } else if (op == "sin") {
      return sin(wrapper->get());
    } else if (op == "cos") {
      return cos(wrapper->get());
    } else if (op == "tan") {
      return tan(wrapper->get());
    } else if (op == "asin") {
      return arcsin(wrapper->get());
    } else if (op == "acos") {
      return arccos(wrapper->get());
    } else if (op == "atan") {
      return arctan(wrapper->get());
    } else if (op == "sinh") {
      return sinh(wrapper->get());
    } else if (op == "cosh") {
      return cosh(wrapper->get());
    } else if (op == "tanh") {
      return tanh(wrapper->get());
    } else if (op == "asinh") {
      return arcsinh(wrapper->get());
    } else if (op == "acosh") {
      return arccosh(wrapper->get());
    } else if (op == "atanh") {
      return arctanh(wrapper->get());
    } else if (op == "floor") {
      return floor(wrapper->get());
    } else if (op == "ceil") {
      return ceil(wrapper->get());
    } else if (op == "round") {
      return round(wrapper->get());
    } else {
      Rcpp::stop("Unsupported unary operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// Binary operations
// [[Rcpp::export]]
SEXP cpp_mlx_binary(SEXP xp1_, SEXP xp2_, std::string op,
                    std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  lhs = astype(lhs, target_dtype);
  rhs = astype(rhs, target_dtype);

  lhs = astype(lhs, target_dtype, target_device);
  rhs = astype(rhs, target_dtype, target_device);

  array result = [&]() -> array {
    if (op == "+") {
      return add(lhs, rhs);
    } else if (op == "-") {
      return subtract(lhs, rhs);
    } else if (op == "*") {
      return multiply(lhs, rhs);
    } else if (op == "/") {
      return divide(lhs, rhs);
    } else if (op == "^") {
      return power(lhs, rhs);
    } else if (op == "==") {
      return equal(lhs, rhs);
    } else if (op == "!=") {
      return not_equal(lhs, rhs);
    } else if (op == "<") {
      return less(lhs, rhs);
    } else if (op == "<=") {
      return less_equal(lhs, rhs);
    } else if (op == ">") {
      return greater(lhs, rhs);
    } else if (op == ">=") {
      return greater_equal(lhs, rhs);
    } else {
      Rcpp::stop("Unsupported binary operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// Reductions
// [[Rcpp::export]]
SEXP cpp_mlx_reduce(SEXP xp_, std::string op) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  array result = [&]() -> array {
    if (op == "sum") {
      return sum(wrapper->get(), false);  // keepdims=false
    } else if (op == "mean") {
      return mean(wrapper->get(), false);
    } else if (op == "min") {
      return min(wrapper->get(), false);
    } else if (op == "max") {
      return max(wrapper->get(), false);
    } else {
      Rcpp::stop("Unsupported reduction operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// Axis reductions
// [[Rcpp::export]]
SEXP cpp_mlx_reduce_axis(SEXP xp_, std::string op, int axis, bool keepdims) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  std::vector<int> axes = {axis};

  array result = [&]() -> array {
    if (op == "sum") {
      return sum(wrapper->get(), axes, keepdims);
    } else if (op == "mean") {
      return mean(wrapper->get(), axes, keepdims);
    } else if (op == "min") {
      return min(wrapper->get(), axes, keepdims);
    } else if (op == "max") {
      return max(wrapper->get(), axes, keepdims);
    } else {
      Rcpp::stop("Unsupported axis reduction operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// Matrix operations
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
SEXP cpp_mlx_matmul(SEXP xp1_, SEXP xp2_,
                    std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* wrapper1 = get_mlx_wrapper(xp1_);
  MlxArrayWrapper* wrapper2 = get_mlx_wrapper(xp2_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);

  array lhs = wrapper1->get();
  array rhs = wrapper2->get();

  lhs = astype(lhs, target_dtype);
  rhs = astype(rhs, target_dtype);

  lhs = astype(lhs, target_dtype, target_device);
  rhs = astype(rhs, target_dtype, target_device);

  array result = matmul(lhs, rhs);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_normal(SEXP dim_, double mean, double std,
                           std::string dtype_str, std::string device_str) {
  IntegerVector dim(dim_);
  Shape shape(dim.begin(), dim.end());

  Dtype dtype = string_to_dtype(dtype_str);
  if (dtype != float32 && dtype != float64) {
    Rcpp::stop("Random normal currently supports dtype = \"float32\" or \"float64\" only.");
  }
  StreamOrDevice dev_input = string_to_device(device_str);
  array result = mlx::core::random::normal(shape, dtype, array(static_cast<float>(mean), dtype), array(static_cast<float>(std), dtype), std::nullopt, dev_input);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_random_bernoulli(SEXP dim_, double prob, std::string device_str) {
  if (prob < 0.0 || prob > 1.0) {
    Rcpp::stop("prob must be between 0 and 1.");
  }
  IntegerVector dim(dim_);
  Shape shape(dim.begin(), dim.end());
  StreamOrDevice dev_input = string_to_device(device_str);
  array result = mlx::core::random::bernoulli(array(static_cast<float>(prob), float32), shape, std::nullopt, dev_input);
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
SEXP cpp_mlx_random_uniform(SEXP dim_, double low, double high,
                            std::string dtype_str, std::string device_str) {
  IntegerVector dim(dim_);
  Shape shape(dim.begin(), dim.end());

  Dtype dtype = string_to_dtype(dtype_str);
  if (dtype != float32 && dtype != float64) {
    Rcpp::stop("Random uniform currently supports dtype = \"float32\" or \"float64\" only.");
  }

  StreamOrDevice dev_input = string_to_device(device_str);
  array result = mlx::core::random::uniform(low, high, shape, dtype, std::nullopt, dev_input);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
void cpp_mlx_synchronize(std::string device_str) {
  Device dev = string_to_device(device_str);
  Stream stream = default_stream(dev);
  synchronize(stream);
}

// Indexing/slicing
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

// Cumulative operations (flatten array)
// [[Rcpp::export]]
SEXP cpp_mlx_cumulative(SEXP xp_, std::string op) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  array arr = wrapper->get();

  array flat = [&]() -> array {
    if (arr.ndim() <= 1) {
      return reshape(arr, Shape{static_cast<int>(arr.size())});
    }

    std::vector<int> perm(arr.ndim());
    std::iota(perm.begin(), perm.end(), 0);
    std::reverse(perm.begin(), perm.end());

    array transposed = transpose(arr, perm);
    transposed = contiguous(transposed);
    return reshape(transposed, Shape{static_cast<int>(arr.size())});
  }();

  array result = [&]() -> array {
    if (op == "cumsum") {
      return cumsum(flat);
    } else if (op == "cumprod") {
      return cumprod(flat);
    } else if (op == "cummax") {
      return cummax(flat);
    } else if (op == "cummin") {
      return cummin(flat);
    } else {
      Rcpp::stop("Unsupported cumulative operation: " + op);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// Linear algebra operations
// [[Rcpp::export]]
SEXP cpp_mlx_solve(SEXP a_xp_, SEXP b_xp_,
                   std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);
  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);

  array result = [&]() -> array {
    if (b_xp_ == R_NilValue) {
      // No b provided: compute matrix inverse
      return linalg::inv(a_cpu, cpu_stream);
    } else {
      // b provided: solve linear system Ax = b
      MlxArrayWrapper* b_wrapper = get_mlx_wrapper(b_xp_);
      array b_cpu = astype(b_wrapper->get(), target_dtype, cpu_stream);
      return linalg::solve(a_cpu, b_cpu, cpu_stream);
    }
  }();

array result_target = astype(result, target_dtype, target_device);

  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_fft(SEXP xp_, bool inverse, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);

  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);

  array input_cpu = astype(wrapper->get(), wrapper->get().dtype(), cpu_stream);
  array result_cpu = inverse ? mlx::core::fft::ifftn(input_cpu, cpu_stream)
                             : mlx::core::fft::fftn(input_cpu, cpu_stream);

  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);

  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_cholesky(SEXP a_xp_, bool upper,
                      std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);

  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);
  array chol_cpu = mlx::core::linalg::cholesky(a_cpu, upper, cpu_stream);
  array chol_target = astype(chol_cpu, target_dtype, target_device);

  return make_mlx_xptr(std::move(chol_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_qr(SEXP a_xp_,
                std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);

  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);
  auto qr_cpu = mlx::core::linalg::qr(a_cpu, cpu_stream);

  array q_target = astype(qr_cpu.first, target_dtype, target_device);
  array r_target = astype(qr_cpu.second, target_dtype, target_device);

  return List::create(
      Named("Q") = make_mlx_xptr(std::move(q_target)),
      Named("R") = make_mlx_xptr(std::move(r_target)));
}

// [[Rcpp::export]]
SEXP cpp_mlx_svd(SEXP a_xp_, bool compute_uv,
                 std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);

  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);
  std::vector<array> svd_cpu = mlx::core::linalg::svd(a_cpu, compute_uv, cpu_stream);

  List out(svd_cpu.size());
  CharacterVector names(svd_cpu.size());
  if (svd_cpu.size() == 3) {
    names[0] = "U";
    names[1] = "S";
    names[2] = "Vh";
  } else if (svd_cpu.size() == 2) {
    names[0] = "S";
    names[1] = "Vh";
  } else if (svd_cpu.size() == 1) {
    names[0] = "S";
  }

  for (size_t i = 0; i < svd_cpu.size(); ++i) {
    array target = astype(svd_cpu[i], target_dtype, target_device);
    out[i] = make_mlx_xptr(std::move(target));
  }

  if (svd_cpu.size() > 0) {
    out.attr("names") = names;
  }

  return out;
}

// [[Rcpp::export]]
SEXP cpp_mlx_pinv(SEXP a_xp_,
                  std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);

  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);
  array pinv_cpu = mlx::core::linalg::pinv(a_cpu, cpu_stream);
  array pinv_target = astype(pinv_cpu, target_dtype, target_device);

  return make_mlx_xptr(std::move(pinv_target));
}
