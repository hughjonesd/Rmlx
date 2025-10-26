// Compilation functions
#include "mlx_helpers.hpp"
#include <mlx/compile.h>
#include <mlx/mlx.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

namespace {

// Extract arrays from R result (single mlx or list of mlx)
std::vector<array> extract_arrays_from_result(SEXP result) {
  std::vector<array> outputs;

  // Case 1: Single mlx object
  if (Rf_inherits(result, "mlx")) {
    List obj(result);
    SEXP ptr = obj["ptr"];
    MlxArrayWrapper* wrapper = get_mlx_wrapper(ptr);
    outputs.push_back(wrapper->get());
    return outputs;
  }

  // Case 2: List of mlx objects
  if (Rf_isVectorList(result)) {
    List res_list(result);
    if (res_list.size() == 0) {
      Rcpp::stop("Compiled function returned empty list. Must return at least one mlx object.");
    }
    for (int i = 0; i < res_list.size(); ++i) {
      SEXP item = res_list[i];
      if (!Rf_inherits(item, "mlx")) {
        Rcpp::stop(
          "Compiled function returned a list with non-mlx element at position %d.\n"
          "All return values must be mlx objects.", i + 1
        );
      }
      List obj(item);
      outputs.push_back(get_mlx_wrapper(obj["ptr"])->get());
    }
    return outputs;
  }

  // Case 3: Invalid return type
  Rcpp::stop(
    "Compiled function must return either:\n"
    "  - A single mlx object, or\n"
    "  - A list of mlx objects\n"
    "Got: %s\n\n"
    "Common mistakes:\n"
    "  - Returning R vectors/matrices (use as_mlx() to convert)\n"
    "  - Returning NULL or other non-mlx values",
    Rf_type2char(TYPEOF(result))
  );
}

} // namespace

// Wrapper to store compiled function
struct CompiledFunctionWrapper {
  Function r_fun;
  std::function<std::vector<array>(const std::vector<array>&)> compiled_fn;

  CompiledFunctionWrapper(Function f, bool shapeless) : r_fun(f) {
    // Create wrapper lambda that calls R function
    auto wrapper = [f](const std::vector<array>& inputs) -> std::vector<array> {
      // Wrap inputs as mlx objects
      List wrapped_inputs(inputs.size());
      for (size_t i = 0; i < inputs.size(); ++i) {
        wrapped_inputs[i] = wrap_array_as_mlx(inputs[i], "gpu");
      }

      // Call R function
      Language call(f);
      for (R_xlen_t i = 0; i < wrapped_inputs.size(); ++i) {
        call.push_back(wrapped_inputs[i]);
      }

      SEXP result;
      try {
        result = call.eval();
      } catch (const eval_error& e) {
        Rcpp::stop(
          "Error evaluating compiled function: %s\n\n"
          "Common issues in compiled functions:\n"
          "  - Printing or inspecting arrays (cannot evaluate placeholders)\n"
          "  - Converting to R with as.matrix(), as.numeric(), [[ extraction\n"
          "  - Control flow based on array values (if/while on array contents)\n"
          "  - Side effects (modifying external variables)\n\n"
          "Compiled functions must be pure and use only MLX operations.",
          e.what()
        );
      }

      // Extract arrays from result
      try {
        return extract_arrays_from_result(result);
      } catch (const std::exception& e) {
        Rcpp::stop("Failed to extract results from compiled function: %s", e.what());
      }
    };

    // Compile the wrapper
    // This returns immediately - tracing happens on first call!
    try {
      compiled_fn = compile(wrapper, shapeless);
    } catch (const std::exception& e) {
      Rcpp::stop("Failed to create compiled function: %s", e.what());
    }
  }
};

// Finalizer for compiled function wrapper
void compiled_fn_finalizer(SEXP xp) {
  if (TYPEOF(xp) == EXTPTRSXP) {
    CompiledFunctionWrapper* wrapper =
      static_cast<CompiledFunctionWrapper*>(R_ExternalPtrAddr(xp));
    if (wrapper != nullptr) {
      delete wrapper;
      R_ClearExternalPtr(xp);
    }
  }
}

// [[Rcpp::export]]
SEXP cpp_mlx_compile_create(SEXP fun_sexp, bool shapeless) {
  if (!Rf_isFunction(fun_sexp)) {
    Rcpp::stop("First argument must be an R function.");
  }

  Function r_fun(fun_sexp);

  // Create wrapper (fast - no tracing yet)
  CompiledFunctionWrapper* wrapper = nullptr;
  try {
    wrapper = new CompiledFunctionWrapper(r_fun, shapeless);
  } catch (const std::exception& e) {
    Rcpp::stop("Failed to create compiled function wrapper: %s", e.what());
  }

  // Store in external pointer
  SEXP xp = R_MakeExternalPtr(wrapper, R_NilValue, R_NilValue);
  R_RegisterCFinalizerEx(xp, compiled_fn_finalizer, TRUE);
  return xp;
}

// [[Rcpp::export]]
List cpp_mlx_compile_call(SEXP compiled_xp, List mlx_args) {
  if (TYPEOF(compiled_xp) != EXTPTRSXP) {
    Rcpp::stop("Expected external pointer to compiled function.");
  }

  CompiledFunctionWrapper* wrapper =
    static_cast<CompiledFunctionWrapper*>(R_ExternalPtrAddr(compiled_xp));

  if (wrapper == nullptr) {
    Rcpp::stop("Invalid compiled function pointer (already finalized?).");
  }

  // Extract arrays from mlx objects
  std::vector<array> inputs;
  std::vector<std::string> devices;
  inputs.reserve(mlx_args.size());
  devices.reserve(mlx_args.size());

  for (int i = 0; i < mlx_args.size(); ++i) {
    List obj(mlx_args[i]);
    if (!obj.inherits("mlx")) {
      Rcpp::stop("Argument %d is not an mlx object. All arguments must be mlx arrays.", i + 1);
    }
    SEXP ptr = obj["ptr"];
    inputs.push_back(get_mlx_wrapper(ptr)->get());
    devices.push_back(as<std::string>(obj["device"]));
  }

  // Call compiled function
  // FIRST call: MLX traces the function (slow)
  // SUBSEQUENT calls: uses cached graph (fast)
  std::vector<array> outputs;
  try {
    outputs = wrapper->compiled_fn(inputs);
  } catch (const std::exception& e) {
    Rcpp::stop(
      "Error calling compiled function: %s\n\n"
      "If this is the first call, the error occurred during tracing.\n"
      "Check that your function uses only MLX operations and doesn't\n"
      "try to inspect array values.",
      e.what()
    );
  }

  // Wrap results as mlx objects
  List result(outputs.size());
  std::string result_device = devices.empty() ? "gpu" : devices[0];
  for (size_t i = 0; i < outputs.size(); ++i) {
    result[i] = wrap_array_as_mlx(outputs[i], result_device);
  }

  return result;
}

// [[Rcpp::export]]
void cpp_mlx_disable_compile() {
  disable_compile();
}

// [[Rcpp::export]]
void cpp_mlx_enable_compile() {
  enable_compile();
}
