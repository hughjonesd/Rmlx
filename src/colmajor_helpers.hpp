// Helpers for handling R (column-major) vs MLX (row-major) layout.
#pragma once

#include <mlx/mlx.h>
#include <vector>

namespace rmlx {

// Reverse axes order; applying twice is identity. Used to swap between
// R's column-major view and MLX's row-major.
inline mlx::core::array transpose_between_mlx_and_r(const mlx::core::array& arr) {
  using namespace mlx::core;
  // MLX transpose with no axes reverses axis order; scalars/vectors are no-ops.
  return transpose(arr);
}

// Flatten an array in R's column-major order into a contiguous 1D vector.
inline mlx::core::array flatten_r_order(const mlx::core::array& arr) {
  using namespace mlx::core;
  if (arr.ndim() <= 1) {
    return reshape(arr, Shape{static_cast<int>(arr.size())});
  }
  array transposed = transpose_between_mlx_and_r(arr);
  transposed = contiguous(transposed);
  return reshape(transposed, Shape{static_cast<int>(transposed.size())});
}

} // namespace rmlx
