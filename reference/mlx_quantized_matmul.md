# Quantized Matrix Multiplication

Performs matrix multiplication with a quantized weight matrix. This
operation is essential for efficient inference with quantized models,
significantly reducing memory usage and computation time while
maintaining reasonable accuracy.

## Usage

``` r
mlx_quantized_matmul(
  x,
  w,
  scales = NULL,
  biases = NULL,
  transpose = TRUE,
  group_size = 64L,
  bits = 4L,
  mode = "affine",
  device = NULL
)
```

## Arguments

- x:

  An mlx array.

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

- transpose:

  Whether to transpose the weight matrix before multiplication.

- group_size:

  The group size for quantization. Smaller groups improve accuracy at
  the cost of slightly higher memory. Default: 64.

- bits:

  Number of bits for quantization (typically 4 or 8). Default: 4.

- mode:

  Quantization mode, either `"affine"` or `"mxfp4"`.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

An mlx array with the result of the quantized matrix multiplication

## Details

Quantized matrix multiplication uses low-precision representations
(typically 4-bit or 8-bit integers) for weights, which reduces memory
footprint by up to 8x compared to float32. The scales parameter contains
the dequantization factors needed to reconstruct approximate float
values during computation.

The group_size parameter controls the granularity of quantization -
smaller groups provide better accuracy but slightly higher memory usage.

**Automatic Quantization**: If only w is provided (without scales), the
function will automatically quantize w using
[`mlx_quantize()`](https://hughjonesd.github.io/Rmlx/reference/mlx_quantize.md)
before performing the multiplication. For repeated operations, it's more
efficient to pre-quantize weights once using
[`mlx_quantize()`](https://hughjonesd.github.io/Rmlx/reference/mlx_quantize.md)
and reuse them.

## See also

[`mlx_quantize()`](https://hughjonesd.github.io/Rmlx/reference/mlx_quantize.md),
[`mlx_dequantize()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dequantize.md),
[`mlx_gather_qmm()`](https://hughjonesd.github.io/Rmlx/reference/mlx_gather_qmm.md)

## Examples

``` r
# Automatic quantization (convenient but slower for repeated use)
x <- mlx_rand_normal(c(4, 64))
w <- mlx_rand_normal(c(128, 64))
result <- mlx_quantized_matmul(x, w, group_size = 32)

# Pre-quantized weights (faster for repeated operations)
quant <- mlx_quantize(w, group_size = 32, bits = 4)
result <- mlx_quantized_matmul(x, quant$w_q, quant$scales, quant$biases, group_size = 32)
```
