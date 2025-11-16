# Gather-based Quantized Matrix Multiplication

Performs quantized matrix multiplication with optional gather operations
on inputs. This is useful for combining embedding lookups with quantized
linear transformations, a common pattern in transformer models.

## Usage

``` r
mlx_gather_qmm(
  x,
  w,
  scales,
  biases = NULL,
  lhs_indices = NULL,
  rhs_indices = NULL,
  transpose = TRUE,
  group_size = 64L,
  bits = 4L,
  mode = "affine",
  sorted_indices = FALSE,
  device = mlx_default_device()
)
```

## Arguments

- x:

  An mlx array.

- w:

  An mlx array (the quantized weight matrix)

- scales:

  An mlx array (the quantization scales)

- biases:

  An optional mlx array (biases to add). Default: NULL

- lhs_indices:

  An optional mlx array (indices for gathering from x). Default: NULL

- rhs_indices:

  An optional mlx array (indices for gathering from w). Default: NULL

- transpose:

  Whether to transpose the weight matrix. Default: TRUE

- group_size:

  The group size for quantization. Default: 64

- bits:

  The number of bits for quantization (typically 4 or 8). Default: 4

- mode:

  The quantization mode, either "affine" or "mxfp4". Default: "affine"

- sorted_indices:

  Whether the indices are sorted (enables optimizations). Default: FALSE

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

An mlx array with the result of the gather-based quantized matrix
multiplication

## Details

This function combines gather operations (indexed lookups) with
quantized matrix multiplication. When lhs_indices is provided, it
performs `x[lhs_indices]` before the multiplication. Similarly,
rhs_indices gathers from the weight matrix.

This is particularly efficient for transformer models where you need to
look up token embeddings and then apply a quantized linear
transformation in one fused operation.

## See also

[`mlx_quantized_matmul()`](https://hughjonesd.github.io/Rmlx/reference/mlx_quantized_matmul.md)

[mlx.nn](https://ml-explore.github.io/mlx/build/html/python/nn.html)
