#include "mlx_bindings.hpp"
#include <mlx/backend/gpu/available.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

// [[Rcpp::export]]
std::string cpp_mlx_default_device() {
  return device_to_string(default_device());
}

// [[Rcpp::export]]
void cpp_mlx_set_default_device(std::string device_str) {
  Device dev = string_to_device(device_str);
  set_default_device(dev);
}

// [[Rcpp::export]]
bool cpp_mlx_has_gpu() {
  return mlx::core::gpu::is_available();
}
