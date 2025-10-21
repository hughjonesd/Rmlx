#include "mlx_bindings.hpp"
#include <mlx/transforms.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

namespace {

List wrap_array_as_mlx(const array& arr, const std::string& device_hint) {
  array copy = arr;
  SEXP ptr = make_mlx_xptr(std::move(copy));

  const Shape& shape = arr.shape();
  IntegerVector dim(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    dim[i] = shape[i];
  }

  std::string dtype = dtype_to_string(arr.dtype());

  List obj = List::create(
      Named("ptr") = ptr,
      Named("dim") = dim,
      Named("dtype") = dtype,
      Named("device") = device_hint);
  obj.attr("class") = "mlx";
  return obj;
}

std::vector<array> extract_primals(const List& args) {
  std::vector<array> primals;
  primals.reserve(args.size());
  for (int i = 0; i < args.size(); ++i) {
    List obj(args[i]);
    SEXP xp = obj["ptr"];
    MlxArrayWrapper* wrapper = get_mlx_wrapper(xp);
    primals.push_back(wrapper->get());
  }
  return primals;
}

} // namespace

// [[Rcpp::export]]
SEXP cpp_mlx_value_grad(SEXP fun_sexp,
                        List args,
                        IntegerVector argnums,
                        bool return_value) {
  if (args.size() == 0) {
    Rcpp::stop("At least one argument is required for differentiation.");
  }

  Function fun(fun_sexp);

  std::vector<std::string> devices(args.size());
  for (int i = 0; i < args.size(); ++i) {
    List obj(args[i]);
    if (!obj.inherits("mlx")) {
      Rcpp::stop("All arguments must be 'mlx' objects. Use as_mlx() to convert.");
    }
    devices[i] = as<std::string>(obj["device"]);
  }

  std::vector<int> argnums_vec(argnums.size());
  for (int i = 0; i < argnums.size(); ++i) {
    int idx = argnums[i];
    if (idx < 0 || idx >= args.size()) {
      Rcpp::stop("argnums must be between 1 and the number of arguments.");
    }
    argnums_vec[i] = idx;
  }

  auto primals = extract_primals(args);

  auto wrap_inputs = [&](const std::vector<array>& inputs) -> List {
    List wrapped(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      wrapped[i] = wrap_array_as_mlx(inputs[i], devices[i]);
    }
    return wrapped;
  };

  auto fun_wrapper =
      [fun, wrap_inputs](const std::vector<array>& inputs) -> array {
    List wrapped_inputs = wrap_inputs(inputs);
    Language call(fun);
    for (R_xlen_t i = 0; i < wrapped_inputs.size(); ++i) {
      call.push_back(wrapped_inputs[i]);
    }

    SEXP result;
    try {
      result = call.eval();
    } catch (const eval_error& e) {
      Rcpp::stop("Error while evaluating function inside mlx_grad(): %s",
                 e.what());
    }

    if (!Rf_inherits(result, "mlx")) {
      Rcpp::stop("Gradient function must return an `mlx` object. "
                 "Ensure your closure keeps computations in MLX or wraps the result with as_mlx().");
    }

    List res_obj(result);
    SEXP res_ptr = res_obj["ptr"];
    MlxArrayWrapper* res_wrap = get_mlx_wrapper(res_ptr);
    return res_wrap->get();
  };

  SimpleValueAndGradFn vg_fn =
      value_and_grad(fun_wrapper, argnums_vec);

  auto vg_result = [&]() {
    try {
      return vg_fn(primals);
    } catch (const std::exception& e) {
      Rcpp::stop("MLX autograd failed to differentiate the function: %s\n"
                 "Ensure all differentiable computations use MLX operations.",
                 e.what());
    }
  }();

  List grads(argnums_vec.size());
  for (size_t i = 0; i < argnums_vec.size(); ++i) {
    int arg_index = argnums_vec[i];
    grads[i] = wrap_array_as_mlx(vg_result.second[i], devices[arg_index]);
  }

  if (!return_value) {
    return grads;
  }

  List value = wrap_array_as_mlx(vg_result.first, devices[0]);
  return List::create(Named("value") = value,
                      Named("grads") = grads);
}

// [[Rcpp::export]]
SEXP cpp_mlx_stop_gradient(SEXP xp_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array stopped = stop_gradient(wrapper->get());
  return make_mlx_xptr(std::move(stopped));
}
