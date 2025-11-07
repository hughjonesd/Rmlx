# 1D Transposed Convolution

Applies a 1D transposed convolution (also called deconvolution) over an
input signal. Transposed convolutions are used to upsample the spatial
dimensions of the input.

## Usage

``` r
mlx_conv_transpose1d(
  input,
  weight,
  stride = 1L,
  padding = 0L,
  dilation = 1L,
  output_padding = 0L,
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

- output_padding:

  Additional size added to output shape. Default: 0

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

An mlx array with the transposed convolution result

## Details

Input has shape `(batch, length, in_channels)` for 'NWC' layout. Weight
has shape `(out_channels, kernel_size, in_channels)`.

## See also

[`mlx_conv1d()`](https://hughjonesd.github.io/Rmlx/reference/mlx_conv1d.md),
[`mlx_conv_transpose2d()`](https://hughjonesd.github.io/Rmlx/reference/mlx_conv_transpose2d.md),
[`mlx_conv_transpose3d()`](https://hughjonesd.github.io/Rmlx/reference/mlx_conv_transpose3d.md)

[mlx.nn](https://ml-explore.github.io/mlx/build/html/python/nn.html)
