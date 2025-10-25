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
