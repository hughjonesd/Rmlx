// Neural network operations
#include "mlx_helpers.hpp"
#include <mlx/mlx.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

// [[Rcpp::export]]
SEXP cpp_mlx_conv1d(SEXP input_xp_, SEXP weight_xp_, int stride, int padding,
                    int dilation, int groups, std::string device_str) {
  MlxArrayWrapper* input_wrapper = get_mlx_wrapper(input_xp_);
  MlxArrayWrapper* weight_wrapper = get_mlx_wrapper(weight_xp_);

  StreamOrDevice target_device = string_to_device(device_str);

  array input = input_wrapper->get();
  array weight = weight_wrapper->get();

  array result = conv1d(input, weight, stride, padding, dilation, groups, target_device);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_conv2d(SEXP input_xp_, SEXP weight_xp_, IntegerVector stride,
                    IntegerVector padding, IntegerVector dilation, int groups,
                    std::string device_str) {
  MlxArrayWrapper* input_wrapper = get_mlx_wrapper(input_xp_);
  MlxArrayWrapper* weight_wrapper = get_mlx_wrapper(weight_xp_);

  StreamOrDevice target_device = string_to_device(device_str);

  array input = input_wrapper->get();
  array weight = weight_wrapper->get();

  std::pair<int, int> stride_pair = {stride[0], stride[1]};
  std::pair<int, int> padding_pair = {padding[0], padding[1]};
  std::pair<int, int> dilation_pair = {dilation[0], dilation[1]};

  array result = conv2d(input, weight, stride_pair, padding_pair, dilation_pair,
                       groups, target_device);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_conv3d(SEXP input_xp_, SEXP weight_xp_, IntegerVector stride,
                    IntegerVector padding, IntegerVector dilation, int groups,
                    std::string device_str) {
  MlxArrayWrapper* input_wrapper = get_mlx_wrapper(input_xp_);
  MlxArrayWrapper* weight_wrapper = get_mlx_wrapper(weight_xp_);

  StreamOrDevice target_device = string_to_device(device_str);

  array input = input_wrapper->get();
  array weight = weight_wrapper->get();

  std::tuple<int, int, int> stride_tuple = {stride[0], stride[1], stride[2]};
  std::tuple<int, int, int> padding_tuple = {padding[0], padding[1], padding[2]};
  std::tuple<int, int, int> dilation_tuple = {dilation[0], dilation[1], dilation[2]};

  array result = conv3d(input, weight, stride_tuple, padding_tuple, dilation_tuple,
                       groups, target_device);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_conv_transpose1d(SEXP input_xp_, SEXP weight_xp_, int stride, int padding,
                               int dilation, int output_padding, int groups,
                               std::string device_str) {
  MlxArrayWrapper* input_wrapper = get_mlx_wrapper(input_xp_);
  MlxArrayWrapper* weight_wrapper = get_mlx_wrapper(weight_xp_);

  StreamOrDevice target_device = string_to_device(device_str);

  array input = input_wrapper->get();
  array weight = weight_wrapper->get();

  array result = conv_transpose1d(input, weight, stride, padding, dilation,
                                   output_padding, groups, target_device);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_conv_transpose2d(SEXP input_xp_, SEXP weight_xp_, IntegerVector stride,
                               IntegerVector padding, IntegerVector dilation,
                               IntegerVector output_padding, int groups,
                               std::string device_str) {
  MlxArrayWrapper* input_wrapper = get_mlx_wrapper(input_xp_);
  MlxArrayWrapper* weight_wrapper = get_mlx_wrapper(weight_xp_);

  StreamOrDevice target_device = string_to_device(device_str);

  array input = input_wrapper->get();
  array weight = weight_wrapper->get();

  std::pair<int, int> stride_pair = {stride[0], stride[1]};
  std::pair<int, int> padding_pair = {padding[0], padding[1]};
  std::pair<int, int> dilation_pair = {dilation[0], dilation[1]};
  std::pair<int, int> output_padding_pair = {output_padding[0], output_padding[1]};

  array result = conv_transpose2d(input, weight, stride_pair, padding_pair,
                                   dilation_pair, output_padding_pair, groups,
                                   target_device);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_conv_transpose3d(SEXP input_xp_, SEXP weight_xp_, IntegerVector stride,
                               IntegerVector padding, IntegerVector dilation,
                               IntegerVector output_padding, int groups,
                               std::string device_str) {
  MlxArrayWrapper* input_wrapper = get_mlx_wrapper(input_xp_);
  MlxArrayWrapper* weight_wrapper = get_mlx_wrapper(weight_xp_);

  StreamOrDevice target_device = string_to_device(device_str);

  array input = input_wrapper->get();
  array weight = weight_wrapper->get();

  std::tuple<int, int, int> stride_tuple = {stride[0], stride[1], stride[2]};
  std::tuple<int, int, int> padding_tuple = {padding[0], padding[1], padding[2]};
  std::tuple<int, int, int> dilation_tuple = {dilation[0], dilation[1], dilation[2]};
  std::tuple<int, int, int> output_padding_tuple = {output_padding[0], output_padding[1], output_padding[2]};

  array result = conv_transpose3d(input, weight, stride_tuple, padding_tuple,
                                   dilation_tuple, output_padding_tuple, groups,
                                   target_device);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_quantized_matmul(SEXP x_xp_, SEXP w_xp_, SEXP scales_xp_,
                               SEXP biases_xp_,
                               bool transpose, int group_size, int bits,
                               std::string mode, std::string device_str) {
  MlxArrayWrapper* x_wrapper = get_mlx_wrapper(x_xp_);
  MlxArrayWrapper* w_wrapper = get_mlx_wrapper(w_xp_);
  MlxArrayWrapper* scales_wrapper = get_mlx_wrapper(scales_xp_);

  StreamOrDevice target_device = string_to_device(device_str);

  array x = x_wrapper->get();
  array w = w_wrapper->get();
  array scales = scales_wrapper->get();

  std::optional<array> biases = std::nullopt;
  if (biases_xp_ != R_NilValue) {
    MlxArrayWrapper* biases_wrapper = get_mlx_wrapper(biases_xp_);
    biases = biases_wrapper->get();
  }

  array result = quantized_matmul(x, w, scales, biases, transpose,
                                   std::optional<int>(group_size),
                                   std::optional<int>(bits), mode, target_device);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_gather_qmm(SEXP x_xp_, SEXP w_xp_, SEXP scales_xp_,
                         SEXP biases_xp_,
                         SEXP lhs_indices_xp_,
                         SEXP rhs_indices_xp_,
                         bool transpose, int group_size, int bits,
                         std::string mode, bool sorted_indices,
                         std::string device_str) {
  MlxArrayWrapper* x_wrapper = get_mlx_wrapper(x_xp_);
  MlxArrayWrapper* w_wrapper = get_mlx_wrapper(w_xp_);
  MlxArrayWrapper* scales_wrapper = get_mlx_wrapper(scales_xp_);

  StreamOrDevice target_device = string_to_device(device_str);

  array x = x_wrapper->get();
  array w = w_wrapper->get();
  array scales = scales_wrapper->get();

  std::optional<array> biases = std::nullopt;
  if (biases_xp_ != R_NilValue) {
    MlxArrayWrapper* biases_wrapper = get_mlx_wrapper(biases_xp_);
    biases = biases_wrapper->get();
  }

  std::optional<array> lhs_indices = std::nullopt;
  if (lhs_indices_xp_ != R_NilValue) {
    MlxArrayWrapper* lhs_wrapper = get_mlx_wrapper(lhs_indices_xp_);
    lhs_indices = lhs_wrapper->get();
  }

  std::optional<array> rhs_indices = std::nullopt;
  if (rhs_indices_xp_ != R_NilValue) {
    MlxArrayWrapper* rhs_wrapper = get_mlx_wrapper(rhs_indices_xp_);
    rhs_indices = rhs_wrapper->get();
  }

  array result = gather_qmm(x, w, scales, biases, lhs_indices, rhs_indices,
                            transpose, std::optional<int>(group_size),
                            std::optional<int>(bits), mode, sorted_indices,
                            target_device);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
List cpp_mlx_quantize(SEXP w_xp_, int group_size, int bits,
                      std::string mode, std::string device_str) {
  MlxArrayWrapper* w_wrapper = get_mlx_wrapper(w_xp_);
  StreamOrDevice target_device = string_to_device(device_str);

  array w = w_wrapper->get();
  std::vector<array> result = quantize(w, std::optional<int>(group_size),
                                       std::optional<int>(bits), mode, target_device);

  // quantize returns a vector with 2 or 3 elements: [w_q, scales, biases (optional)]
  List out;
  out["w_q"] = make_mlx_xptr(std::move(result[0]));
  out["scales"] = make_mlx_xptr(std::move(result[1]));
  if (result.size() > 2) {
    out["biases"] = make_mlx_xptr(std::move(result[2]));
  } else {
    out["biases"] = R_NilValue;
  }

  return out;
}

// [[Rcpp::export]]
SEXP cpp_mlx_dequantize(SEXP w_xp_, SEXP scales_xp_, SEXP biases_xp_,
                        int group_size, int bits, std::string mode,
                        std::string device_str) {
  MlxArrayWrapper* w_wrapper = get_mlx_wrapper(w_xp_);
  MlxArrayWrapper* scales_wrapper = get_mlx_wrapper(scales_xp_);

  StreamOrDevice target_device = string_to_device(device_str);

  array w = w_wrapper->get();
  array scales = scales_wrapper->get();

  std::optional<array> biases = std::nullopt;
  if (biases_xp_ != R_NilValue) {
    MlxArrayWrapper* biases_wrapper = get_mlx_wrapper(biases_xp_);
    biases = biases_wrapper->get();
  }

  array result = dequantize(w, scales, biases, std::optional<int>(group_size),
                           std::optional<int>(bits), mode, std::nullopt, target_device);

  return make_mlx_xptr(std::move(result));
}
