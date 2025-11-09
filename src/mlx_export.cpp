#include "mlx_bindings.hpp"
#include "mlx_helpers.hpp"
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

// [[Rcpp::export]]
SEXP cpp_mlx_import_function(const std::string& path) {
  ImportedFunction function = import_function(path);
  return make_mlx_imported_function_xptr(std::move(function));
}

// [[Rcpp::export]]
SEXP cpp_mlx_call_imported(SEXP fn_xp,
                           List args_ptrs,
                           List kwargs_ptrs,
                           std::string device_str) {
  MlxImportedFunctionWrapper* fn_wrapper = get_mlx_imported_function(fn_xp);
  const ImportedFunction& function = fn_wrapper->get();

  Args positional;
  positional.reserve(args_ptrs.size());
  for (int i = 0; i < args_ptrs.size(); ++i) {
    SEXP ptr = args_ptrs[i];
    if (ptr == R_NilValue) {
      continue;
    }
    MlxArrayWrapper* wrapper = get_mlx_wrapper(ptr);
    positional.push_back(wrapper->get());
  }

  Kwargs kwargs;
  SEXP names_attr = kwargs_ptrs.attr("names");
  CharacterVector names;
  if (kwargs_ptrs.size() > 0) {
    if (names_attr == R_NilValue) {
      stop("Keyword arguments must be named.");
    }
    names = CharacterVector(names_attr);
  }
  for (int i = 0; i < kwargs_ptrs.size(); ++i) {
    SEXP ptr = kwargs_ptrs[i];
    if (ptr == R_NilValue) {
      continue;
    }
    if (names[i] == NA_STRING) {
      stop("Keyword arguments must be named.");
    }
    std::string key = as<std::string>(names[i]);
    MlxArrayWrapper* wrapper = get_mlx_wrapper(ptr);
    kwargs.emplace(std::move(key), wrapper->get());
  }

  std::vector<array> result;
  if (!positional.empty() && !kwargs.empty()) {
    result = function(positional, kwargs);
  } else if (!kwargs.empty()) {
    result = function(kwargs);
  } else {
    result = function(positional);
  }

  StreamOrDevice target_device = string_to_device(device_str);
  List out(result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    array converted = astype(result[i], result[i].dtype(), target_device);
    out[i] = wrap_array_as_mlx(converted, device_str);
  }

  return out;
}
