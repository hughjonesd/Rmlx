# Quantization parameters

Quantization parameters

## Arguments

- w:

  An mlx array representing the weight matrix. Accepts either an
  unquantized matrix (which may be quantized automatically) or a
  pre-quantized uint32 matrix produced by
  [`mlx_quantize()`](https://hughjonesd.github.io/Rmlx/reference/mlx_quantize.md).

- scales:

  An optional mlx array of quantization scales. Required when `w` is
  already quantized.

- biases:

  An optional mlx array of quantization biases (affine mode); use `NULL`
  for symmetric quantization.

- group_size:

  The group size for quantization. Smaller groups improve accuracy at
  the cost of slightly higher memory. Default: 64.

- bits:

  Number of bits for quantization (typically 4 or 8). Default: 4.

- mode:

  Quantization mode, either `"affine"` or `"mxfp4"`.

- transpose:

  Whether to transpose the weight matrix before multiplication.

- sorted_indices:

  Whether supplied indices are sorted (enables optimizations in
  gather-based kernels).
