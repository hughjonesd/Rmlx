#include "mlx_helpers.hpp"
#include <Rcpp.h>
#include <mlx/mlx.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

// [[Rcpp::export]]
SEXP cpp_mlx_stream_new(std::string device_str) {
  Device device = string_to_device(device_str);
  Stream stream = new_stream(device);
  return make_mlx_stream_xptr(stream);
}

// [[Rcpp::export]]
SEXP cpp_mlx_stream_default(std::string device_str) {
  Device device = string_to_device(device_str);
  Stream stream = default_stream(device);
  return make_mlx_stream_xptr(stream);
}

// [[Rcpp::export]]
void cpp_mlx_set_default_stream(SEXP stream_xp) {
  Stream stream = get_mlx_stream(stream_xp);
  set_default_stream(stream);
}

// [[Rcpp::export]]
std::string cpp_mlx_stream_device(SEXP stream_xp) {
  Stream stream = get_mlx_stream(stream_xp);
  return device_to_string(stream.device);
}

// [[Rcpp::export]]
int cpp_mlx_stream_index(SEXP stream_xp) {
  Stream stream = get_mlx_stream(stream_xp);
  return stream.index;
}

// [[Rcpp::export]]
void cpp_mlx_synchronize_stream(SEXP stream_xp) {
  Stream stream = get_mlx_stream(stream_xp);
  synchronize(stream);
}
