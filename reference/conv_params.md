# Common Convolution Parameters

Common Convolution Parameters

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
