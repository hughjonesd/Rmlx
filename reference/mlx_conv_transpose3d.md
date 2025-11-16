# 3D Transposed Convolution

Applies a 3D transposed convolution (also called deconvolution) over an
input signal. Useful for 3D volumetric data upsampling, such as in
medical imaging or video generation.

## Usage

``` r
mlx_conv_transpose3d(
  input,
  weight,
  stride = c(1L, 1L, 1L),
  padding = c(0L, 0L, 0L),
  dilation = c(1L, 1L, 1L),
  output_padding = c(0L, 0L, 0L),
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

  Additional size added to output shape. Can be a scalar or length-3
  vector. Default: c(0, 0, 0)

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

An mlx array with the transposed convolution result

## Details

Input has shape `(batch, depth, height, width, in_channels)` for 'NDHWC'
layout. Weight has shape
`(out_channels, kernel_d, kernel_h, kernel_w, in_channels)`.

## See also

[`mlx_conv3d()`](https://hughjonesd.github.io/Rmlx/reference/mlx_conv3d.md),
[`mlx_conv_transpose1d()`](https://hughjonesd.github.io/Rmlx/reference/mlx_conv_transpose1d.md),
[`mlx_conv_transpose2d()`](https://hughjonesd.github.io/Rmlx/reference/mlx_conv_transpose2d.md)

[mlx.nn](https://ml-explore.github.io/mlx/build/html/python/nn.html)
