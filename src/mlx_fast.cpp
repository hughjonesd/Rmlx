// Fast/custom kernel operations
#include "mlx_helpers.hpp"
#include <mlx/fast.h>
#include <Rcpp.h>
#include <cmath>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;
using namespace mlx::core::fast;

namespace {

struct MetalKernelWrapper {
  CustomKernelFunction kernel;
  std::vector<std::string> output_names;

  MetalKernelWrapper(const std::string& name,
                     const std::vector<std::string>& input_names,
                     const std::vector<std::string>& output_names_,
                     const std::string& source,
                     const std::string& header,
                     bool ensure_row_contiguous,
                     bool atomic_outputs)
    : kernel(metal_kernel(
        name,
        input_names,
        output_names_,
        source,
        header,
        ensure_row_contiguous,
        atomic_outputs
      )),
      output_names(output_names_) {}
};

void metal_kernel_finalizer(SEXP xp) {
  if (TYPEOF(xp) == EXTPTRSXP) {
    MetalKernelWrapper* wrapper =
      static_cast<MetalKernelWrapper*>(R_ExternalPtrAddr(xp));
    if (wrapper != nullptr) {
      delete wrapper;
      R_ClearExternalPtr(xp);
    }
  }
}

std::tuple<int, int, int> parse_launch_dims(const IntegerVector& dims,
                                            const char* arg_name) {
  if (dims.size() != 3) {
    Rcpp::stop("%s must have length 3.", arg_name);
  }
  if (is_true(any(is_na(dims)))) {
    Rcpp::stop("%s cannot contain NA values.", arg_name);
  }
  for (int i = 0; i < dims.size(); ++i) {
    if (dims[i] <= 0) {
      Rcpp::stop("%s must contain positive integers.", arg_name);
    }
  }
  return std::make_tuple(dims[0], dims[1], dims[2]);
}

std::vector<Shape> parse_output_shapes(const List& output_shapes) {
  if (output_shapes.size() == 0) {
    Rcpp::stop("output_shapes must contain at least one shape.");
  }

  std::vector<Shape> result;
  result.reserve(output_shapes.size());
  for (int i = 0; i < output_shapes.size(); ++i) {
    IntegerVector shape_vec(output_shapes[i]);
    if (shape_vec.size() == 0) {
      Rcpp::stop("output shape %d must contain at least one dimension.", i + 1);
    }
    if (is_true(any(is_na(shape_vec)))) {
      Rcpp::stop("output shape %d cannot contain NA values.", i + 1);
    }

    Shape shape;
    shape.reserve(shape_vec.size());
    for (int dim : shape_vec) {
      if (dim <= 0) {
        Rcpp::stop("output shape %d must contain positive integers.", i + 1);
      }
      shape.push_back(dim);
    }
    result.push_back(std::move(shape));
  }
  return result;
}

std::vector<Dtype> parse_output_dtypes(const CharacterVector& output_dtypes) {
  if (output_dtypes.size() == 0) {
    Rcpp::stop("output_dtypes must contain at least one dtype.");
  }
  if (is_true(any(is_na(output_dtypes)))) {
    Rcpp::stop("output_dtypes cannot contain NA values.");
  }

  std::vector<Dtype> result;
  result.reserve(output_dtypes.size());
  for (const auto& dtype : output_dtypes) {
    result.push_back(string_to_dtype(as<std::string>(dtype)));
  }
  return result;
}

TemplateArg parse_template_value(SEXP value, const std::string& name) {
  if (TYPEOF(value) == LGLSXP) {
    LogicalVector vec(value);
    if (vec.size() != 1 || LogicalVector::is_na(vec[0])) {
      Rcpp::stop("Template value '%s' must be a non-missing scalar.", name);
    }
    return static_cast<bool>(vec[0]);
  }

  if (TYPEOF(value) == INTSXP) {
    IntegerVector vec(value);
    if (vec.size() != 1 || IntegerVector::is_na(vec[0])) {
      Rcpp::stop("Template value '%s' must be a non-missing scalar.", name);
    }
    return static_cast<int>(vec[0]);
  }

  if (TYPEOF(value) == REALSXP) {
    NumericVector vec(value);
    if (vec.size() != 1 || NumericVector::is_na(vec[0])) {
      Rcpp::stop("Template value '%s' must be a non-missing scalar.", name);
    }
    double raw = vec[0];
    if (std::floor(raw) != raw) {
      Rcpp::stop("Numeric template value '%s' must be a whole number.", name);
    }
    return static_cast<int>(raw);
  }

  if (TYPEOF(value) == STRSXP) {
    CharacterVector vec(value);
    if (vec.size() != 1 || CharacterVector::is_na(vec[0])) {
      Rcpp::stop("Template value '%s' must be a non-missing scalar.", name);
    }
    return string_to_dtype(as<std::string>(vec[0]));
  }

  Rcpp::stop(
    "Template value '%s' must be logical, integer, numeric, or dtype string.",
    name
  );
}

std::vector<std::pair<std::string, TemplateArg>> parse_template_args(
    const List& template_args) {
  CharacterVector names = template_args.names();
  if (template_args.size() > 0 && names.size() != template_args.size()) {
    Rcpp::stop("template must be a named list.");
  }

  std::vector<std::pair<std::string, TemplateArg>> result;
  result.reserve(template_args.size());
  for (int i = 0; i < template_args.size(); ++i) {
    std::string name = as<std::string>(names[i]);
    if (name.empty()) {
      Rcpp::stop("template must be a named list.");
    }
    result.emplace_back(name, parse_template_value(template_args[i], name));
  }
  return result;
}

} // namespace

// [[Rcpp::export]]
SEXP cpp_mlx_metal_kernel_create(std::string name,
                                 CharacterVector input_names,
                                 CharacterVector output_names,
                                 std::string source,
                                 std::string header,
                                 bool ensure_row_contiguous,
                                 bool atomic_outputs) {
  if (input_names.size() == 0) {
    Rcpp::stop("input_names must contain at least one name.");
  }
  if (output_names.size() == 0) {
    Rcpp::stop("output_names must contain at least one name.");
  }

  std::vector<std::string> inputs =
    as<std::vector<std::string>>(input_names);
  std::vector<std::string> outputs =
    as<std::vector<std::string>>(output_names);

  MetalKernelWrapper* wrapper = nullptr;
  try {
    wrapper = new MetalKernelWrapper(
      name,
      inputs,
      outputs,
      source,
      header,
      ensure_row_contiguous,
      atomic_outputs
    );
  } catch (const std::exception& e) {
    Rcpp::stop("Failed to create Metal kernel: %s", e.what());
  }

  SEXP xp = R_MakeExternalPtr(wrapper, R_NilValue, R_NilValue);
  R_RegisterCFinalizerEx(xp, metal_kernel_finalizer, TRUE);
  return xp;
}

// [[Rcpp::export]]
List cpp_mlx_metal_kernel_call(SEXP kernel_xp,
                               List mlx_args,
                               List output_shapes,
                               CharacterVector output_dtypes,
                               IntegerVector grid,
                               IntegerVector threadgroup,
                               List template_args,
                               SEXP init_value_,
                               bool verbose) {
  if (TYPEOF(kernel_xp) != EXTPTRSXP) {
    Rcpp::stop("Expected external pointer to Metal kernel.");
  }

  Device device = default_device();
  if (device_to_string(device) != "gpu") {
    Rcpp::stop("Metal kernels currently require the gpu device.");
  }

  MetalKernelWrapper* wrapper =
    static_cast<MetalKernelWrapper*>(R_ExternalPtrAddr(kernel_xp));
  if (wrapper == nullptr) {
    Rcpp::stop("Invalid Metal kernel pointer (already finalized?).");
  }

  std::vector<array> inputs;
  inputs.reserve(mlx_args.size());
  for (int i = 0; i < mlx_args.size(); ++i) {
    List obj(mlx_args[i]);
    if (!obj.inherits("mlx")) {
      Rcpp::stop("Argument %d is not an mlx object.", i + 1);
    }
    inputs.push_back(get_mlx_wrapper(obj["ptr"])->get());
  }

  std::vector<Shape> shapes = parse_output_shapes(output_shapes);
  std::vector<Dtype> dtypes = parse_output_dtypes(output_dtypes);
  if (shapes.size() != dtypes.size()) {
    Rcpp::stop(
      "output_shapes and output_dtypes must have the same length."
    );
  }

  std::optional<float> init_value = std::nullopt;
  if (!Rf_isNull(init_value_)) {
    NumericVector init_vec(init_value_);
    if (init_vec.size() != 1 || NumericVector::is_na(init_vec[0])) {
      Rcpp::stop("init_value must be NULL or a non-missing numeric scalar.");
    }
    init_value = static_cast<float>(init_vec[0]);
  }

  std::vector<std::pair<std::string, TemplateArg>> template_vec =
    parse_template_args(template_args);

  std::vector<array> outputs;
  try {
    outputs = wrapper->kernel(
      inputs,
      shapes,
      dtypes,
      parse_launch_dims(grid, "grid"),
      parse_launch_dims(threadgroup, "threadgroup"),
      template_vec,
      init_value,
      verbose,
      device
    );
  } catch (const std::exception& e) {
    Rcpp::stop("Metal kernel execution failed: %s", e.what());
  }

  List result(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    result[i] = wrap_array_as_mlx(outputs[i]);
  }

  if (wrapper->output_names.size() == outputs.size()) {
    CharacterVector names(wrapper->output_names.size());
    for (size_t i = 0; i < wrapper->output_names.size(); ++i) {
      names[i] = wrapper->output_names[i];
    }
    result.names() = names;
  }

  return result;
}
