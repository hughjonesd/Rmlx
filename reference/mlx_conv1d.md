# 1D Convolution

Applies a 1D convolution over the input signal.

## Usage

``` r
mlx_conv1d(
  input,
  weight,
  stride = 1L,
  padding = 0L,
  dilation = 1L,
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
  Default:
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

Convolved output array

## Details

Input has shape `(N, L, C_in)` where N is batch size, L is sequence
length, and C_in is number of input channels. Weight has shape
`(C_out, kernel_size, C_in)`.

## See also

[mlx.core.conv1d](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv1d.html)
