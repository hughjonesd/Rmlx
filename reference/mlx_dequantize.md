# Dequantize a Matrix

Reconstructs an approximate floating-point matrix from a quantized
representation produced by
[`mlx_quantize()`](https://hughjonesd.github.io/Rmlx/reference/mlx_quantize.md).

## Usage

``` r
mlx_dequantize(
  w,
  scales,
  biases = NULL,
  group_size = 64L,
  bits = 4L,
  mode = "affine",
  device = NULL
)
```

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

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

An mlx array with the dequantized (approximate) floating-point weights

## Details

Dequantization unpacks the low-precision quantized weights and applies
the scales (and biases) to reconstruct approximate floating-point
values. Note that some precision is lost during quantization and cannot
be recovered.

## See also

[`mlx_quantize()`](https://hughjonesd.github.io/Rmlx/reference/mlx_quantize.md),
[`mlx_quantized_matmul()`](https://hughjonesd.github.io/Rmlx/reference/mlx_quantized_matmul.md)

## Examples

``` r
w <- mlx_rand_normal(c(64, 32))
quant <- mlx_quantize(w, group_size = 32)
w_reconstructed <- mlx_dequantize(quant$w_q, quant$scales, quant$biases, group_size = 32)
```
