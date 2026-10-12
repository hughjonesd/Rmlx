// Neural network operations
#include "mlx_helpers.hpp"
#include <mlx/mlx.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

// [[Rcpp::export]]
SEXP cpp_mlx_conv1d(SEXP input_xp_, SEXP weight_xp_, int stride, int padding,
                    int dilation, int groups) {
  array input = get_mlx_array(input_xp_);
  array weight = get_mlx_array(weight_xp_);

  array result = conv1d(input, weight, stride, padding, dilation, groups);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_conv2d(SEXP input_xp_, SEXP weight_xp_, IntegerVector stride,
                    IntegerVector padding, IntegerVector dilation, int groups) {
  array input = get_mlx_array(input_xp_);
  array weight = get_mlx_array(weight_xp_);

  std::pair<int, int> stride_pair = {stride[0], stride[1]};
  std::pair<int, int> padding_pair = {padding[0], padding[1]};
  std::pair<int, int> dilation_pair = {dilation[0], dilation[1]};

  array result = conv2d(input, weight, stride_pair, padding_pair, dilation_pair,
                       groups);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_conv3d(SEXP input_xp_, SEXP weight_xp_, IntegerVector stride,
                    IntegerVector padding, IntegerVector dilation, int groups) {
  array input = get_mlx_array(input_xp_);
  array weight = get_mlx_array(weight_xp_);

  std::tuple<int, int, int> stride_tuple = {stride[0], stride[1], stride[2]};
  std::tuple<int, int, int> padding_tuple = {padding[0], padding[1], padding[2]};
  std::tuple<int, int, int> dilation_tuple = {dilation[0], dilation[1], dilation[2]};

  array result = conv3d(input, weight, stride_tuple, padding_tuple, dilation_tuple,
                       groups);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_conv_transpose1d(SEXP input_xp_, SEXP weight_xp_, int stride, int padding,
                               int dilation, int output_padding, int groups) {
  array input = get_mlx_array(input_xp_);
  array weight = get_mlx_array(weight_xp_);

  array result = conv_transpose1d(input, weight, stride, padding, dilation,
                                   output_padding, groups);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_conv_transpose2d(SEXP input_xp_, SEXP weight_xp_, IntegerVector stride,
                               IntegerVector padding, IntegerVector dilation,
                               IntegerVector output_padding, int groups) {
  array input = get_mlx_array(input_xp_);
  array weight = get_mlx_array(weight_xp_);

  std::pair<int, int> stride_pair = {stride[0], stride[1]};
  std::pair<int, int> padding_pair = {padding[0], padding[1]};
  std::pair<int, int> dilation_pair = {dilation[0], dilation[1]};
  std::pair<int, int> output_padding_pair = {output_padding[0], output_padding[1]};

  array result = conv_transpose2d(input, weight, stride_pair, padding_pair,
                                   dilation_pair, output_padding_pair, groups);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_conv_transpose3d(SEXP input_xp_, SEXP weight_xp_, IntegerVector stride,
                               IntegerVector padding, IntegerVector dilation,
                               IntegerVector output_padding, int groups) {
  array input = get_mlx_array(input_xp_);
  array weight = get_mlx_array(weight_xp_);

  std::tuple<int, int, int> stride_tuple = {stride[0], stride[1], stride[2]};
  std::tuple<int, int, int> padding_tuple = {padding[0], padding[1], padding[2]};
  std::tuple<int, int, int> dilation_tuple = {dilation[0], dilation[1], dilation[2]};
  std::tuple<int, int, int> output_padding_tuple = {output_padding[0], output_padding[1], output_padding[2]};

  array result = conv_transpose3d(input, weight, stride_tuple, padding_tuple,
                                   dilation_tuple, output_padding_tuple, groups);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_quantized_matmul(SEXP x_xp_, SEXP w_xp_, SEXP scales_xp_,
                               SEXP biases_xp_,
                               bool transpose, int group_size, int bits,
                               std::string mode) {
  array x = get_mlx_array(x_xp_);
  array w = get_mlx_array(w_xp_);
  array scales = get_mlx_array(scales_xp_);

  std::optional<array> biases = std::nullopt;
  if (biases_xp_ != R_NilValue) {
    array& biases_ref = get_mlx_array(biases_xp_);
    biases = biases_ref;
  }

  array result = quantized_matmul(x, w, scales, biases, transpose,
                                   std::optional<int>(group_size),
                                   std::optional<int>(bits), mode);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_gather_qmm(SEXP x_xp_, SEXP w_xp_, SEXP scales_xp_,
                         SEXP biases_xp_,
                         SEXP lhs_indices_xp_,
                         SEXP rhs_indices_xp_,
                         bool transpose, int group_size, int bits,
                         std::string mode, bool sorted_indices) {
  array x = get_mlx_array(x_xp_);
  array w = get_mlx_array(w_xp_);
  array scales = get_mlx_array(scales_xp_);

  std::optional<array> biases = std::nullopt;
  if (biases_xp_ != R_NilValue) {
    array& biases_ref = get_mlx_array(biases_xp_);
    biases = biases_ref;
  }

  std::optional<array> lhs_indices = std::nullopt;
  if (lhs_indices_xp_ != R_NilValue) {
    array& lhs_ref = get_mlx_array(lhs_indices_xp_);
    lhs_indices = lhs_ref;
  }

  std::optional<array> rhs_indices = std::nullopt;
  if (rhs_indices_xp_ != R_NilValue) {
    array& rhs_ref = get_mlx_array(rhs_indices_xp_);
    rhs_indices = rhs_ref;
  }

  array result = gather_qmm(x, w, scales, biases, lhs_indices, rhs_indices,
                            transpose, std::optional<int>(group_size),
                            std::optional<int>(bits), mode, std::nullopt, sorted_indices);

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
List cpp_mlx_quantize(SEXP w_xp_, int group_size, int bits,
                      std::string mode) {
  array w = get_mlx_array(w_xp_);

  std::vector<array> result = quantize(w, std::optional<int>(group_size),
                                       std::optional<int>(bits), mode,
                                       std::nullopt);

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
                        int group_size, int bits, std::string mode) {
  array w = get_mlx_array(w_xp_);
  array scales = get_mlx_array(scales_xp_);

  std::optional<array> biases = std::nullopt;
  if (biases_xp_ != R_NilValue) {
    array& biases_ref = get_mlx_array(biases_xp_);
    biases = biases_ref;
  }

  array result = dequantize(w, scales, biases, std::optional<int>(group_size),
                            std::optional<int>(bits), mode, std::nullopt,
                            std::nullopt);

  return make_mlx_xptr(std::move(result));
}
