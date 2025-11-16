# 2D Convolution

Applies a 2D convolution over the input image.

## Usage

``` r
mlx_conv2d(
  input,
  weight,
  stride = c(1L, 1L),
  padding = c(0L, 0L),
  dilation = c(1L, 1L),
  groups = 1L,
  device = mlx_default_device()
)
```

## Arguments

- input:

  Input mlx array. Shape depends on dimensionality (see individual
  functions).

- weight:

  Weight array. Shape depends on dimensionality (see individual
  functions).

- stride:

  Stride of the convolution. Can be a scalar or vector (length depends
  on dimensionality). Default: 1 for 1D, c(1,1) for 2D, c(1,1,1) for 3D.

- padding:

  Amount of zero padding. Can be a scalar or vector (length depends on
  dimensionality). Default: 0 for 1D, c(0,0) for 2D, c(0,0,0) for 3D.

- dilation:

  Spacing between kernel elements. Can be a scalar or vector (length
  depends on dimensionality). Default: 1 for 1D, c(1,1) for 2D, c(1,1,1)
  for 3D.

- groups:

  Number of blocked connections from input to output channels. Default:
  1.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

Convolved output array

## Details

Input has shape `(N, H, W, C_in)` where N is batch size, H and W are
height and width, and C_in is number of input channels. Weight has shape
`(C_out, kernel_h, kernel_w, C_in)`.

## See also

[mlx.core.conv2d](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv2d.html)

## Examples

``` r
# Create a simple 2D convolution
input <- as_mlx(array(rnorm(1*28*28*3), dim = c(1, 28, 28, 3)))  # Batch of 1 RGB image
weight <- as_mlx(array(rnorm(16*3*3*3), dim = c(16, 3, 3, 3)))  # 16 filters, 3x3 kernel
output <- mlx_conv2d(input, weight, stride = c(1, 1), padding = c(1, 1))
```
