# Quantize a Matrix

Quantizes a weight matrix to low-precision representation (typically
4-bit or 8-bit). This reduces memory usage and enables faster
computation during inference.

## Usage

``` r
mlx_quantize(
  w,
  group_size = 64L,
  bits = 4L,
  mode = "affine",
  device = mlx_default_device()
)
```

## Arguments

- w:

  An mlx array (the weight matrix to quantize)

- group_size:

  The group size for quantization. Smaller groups provide better
  accuracy but slightly higher memory. Default: 64

- bits:

  The number of bits for quantization (typically 4 or 8). Default: 4

- mode:

  The quantization mode: "affine" (with scales and biases) or "mxfp4"
  (4-bit floating point with group_size=32). Default: "affine"

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

A list containing:

- w_q:

  The quantized weight matrix (packed as uint32)

- scales:

  The quantization scales for dequantization

- biases:

  The quantization biases (NULL for symmetric mode)

## Details

Quantization converts floating-point weights to low-precision integers,
reducing memory by up to 8x for 4-bit quantization. The scales (and
optionally biases) are stored to enable approximate reconstruction of
the original values.

## See also

[`mlx_dequantize()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dequantize.md),
[`mlx_quantized_matmul()`](https://hughjonesd.github.io/Rmlx/reference/mlx_quantized_matmul.md)

## Examples

``` r
w <- mlx_rand_normal(c(64, 32))
quant <- mlx_quantize(w, group_size = 32, bits = 4)
# Use quant$w_q, quant$scales, quant$biases with mlx_quantized_matmul()
```
