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
  device = mlx_default_device()
)
```

## Arguments

- w:

  An mlx array (the quantized weight matrix)

- scales:

  An mlx array (the quantization scales)

- biases:

  An optional mlx array (the quantization biases for affine mode).
  Default: NULL

- group_size:

  The group size used during quantization. Default: 64

- bits:

  The number of bits used during quantization. Default: 4

- mode:

  The quantization mode used: "affine" or "mxfp4". Default: "affine"

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Default:
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

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
if (FALSE) { # \dontrun{
w <- mlx_random_normal(c(512, 256))
quant <- mlx_quantize(w)
w_reconstructed <- mlx_dequantize(quant$w_q, quant$scales, quant$biases)
} # }
```
