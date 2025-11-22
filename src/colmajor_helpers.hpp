// Helpers for handling R (column-major) vs MLX (row-major) layout.
#pragma once

#include <mlx/mlx.h>
#include <vector>

namespace rmlx {

// Reverse axes to go between R column-major view and MLX row-major.
inline mlx::core::array transpose_to_r_order(const mlx::core::array& arr) {
  using namespace mlx::core;
  if (arr.ndim() <= 1) return arr;
  std::vector<int> perm(arr.ndim());
  for (size_t i = 0; i < perm.size(); ++i) {
    perm[i] = static_cast<int>(perm.size() - 1 - i);
  }
  return transpose(arr, perm);
}

// Flatten an array in R's column-major order into a contiguous 1D vector.
inline mlx::core::array flatten_r_order(const mlx::core::array& arr) {
  using namespace mlx::core;
  if (arr.ndim() <= 1) {
    return reshape(arr, Shape{static_cast<int>(arr.size())});
  }
  array transposed = transpose_to_r_order(arr);
  transposed = contiguous(transposed);
  return reshape(transposed, Shape{static_cast<int>(transposed.size())});
}

} // namespace rmlx

