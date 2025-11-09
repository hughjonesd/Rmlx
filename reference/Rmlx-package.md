# Rmlx: R Interface to Apple's MLX Arrays

This package provides an R interface to Apple's
[MLX](https://mlx-framework.org/) (Machine Learning eXchange) library
for GPU-accelerated array operations on Apple Silicon.

## Key Features

- Lazy evaluation: Operations are not computed until explicitly
  evaluated

- GPU acceleration: Leverage Metal on Apple Silicon

- Familiar syntax: S3 methods for standard R operations

- Unified memory: Efficient data sharing between CPU and GPU

## Lazy Evaluation

MLX arrays use lazy evaluation by default. Operations are recorded but
not executed until:

- You call
  [`mlx_eval()`](https://hughjonesd.github.io/Rmlx/reference/mlx_eval.md)

- You convert to R with
  [`as.matrix()`](https://rdrr.io/r/base/matrix.html) or
  [`as.vector()`](https://rdrr.io/r/base/vector.html)

- The result is needed for another computation

The package implements most of the C++ API via calls with the `mlx_`
prefix, but it also ships S3 methods (prefixed `mlx_` or attached to
base generics) so common R matrix operations continue to work on MLX
arrays. R conventions are used throughout: for example, indexing is
1-based.

## See also

Useful links:

- <https://hughjonesd.github.io/Rmlx/>

- <https://github.com/hughjonesd/Rmlx>

- Report bugs at <https://github.com/hughjonesd/Rmlx/issues>

## Author

**Maintainer**: David Hugh-Jones <david@hughjones.com>
