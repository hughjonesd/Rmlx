# Changelog

## Rmlx 0.1.0.9000 (development)

- [`mlx_slice_update()`](https://hughjonesd.github.io/Rmlx/reference/mlx_slice_update.md)
  now accepts 1-based (inclusive) `start`/`stop` indices to match the
  rest of the R API; internal callers were updated accordingly.
- Added
  [`mlx_shape()`](https://hughjonesd.github.io/Rmlx/reference/dim.mlx.md)
  to expose MLX-native shapes (never `NULL`) and aligned
  [`dim.mlx()`](https://hughjonesd.github.io/Rmlx/reference/dim.mlx.md)
  with base R semantics (returns `NULL` for 1-D vectors/scalars).
- [`mlx_rand_categorical()`](https://hughjonesd.github.io/Rmlx/reference/mlx_rand_categorical.md),
  [`mlx_rand_permutation()`](https://hughjonesd.github.io/Rmlx/reference/mlx_rand_permutation.md),
  [`mlx_cross_entropy()`](https://hughjonesd.github.io/Rmlx/reference/mlx_cross_entropy.md),
  and
  [`mlx_embedding()`](https://hughjonesd.github.io/Rmlx/reference/mlx_embedding.md)
  now accept 1-based indices for inputs/outputs, keeping all exported
  APIs consistent with R conventions.
- Added negative numeric indexing support for `[`/`[<-` on `mlx` arrays
  and documented subsetting semantics.
- Added
  [`mlx_import_function()`](https://hughjonesd.github.io/Rmlx/reference/mlx_import_function.md)
  to import MLX functions from (e.g.) Python.
- Added
  [`mlx_array()`](https://hughjonesd.github.io/Rmlx/reference/mlx_array.md),
  [`mlx_matrix()`](https://hughjonesd.github.io/Rmlx/reference/mlx_matrix.md),
  [`mlx_vector()`](https://hughjonesd.github.io/Rmlx/reference/mlx_vector.md),
  and
  [`mlx_scalar()`](https://hughjonesd.github.io/Rmlx/reference/mlx_scalar.md)
  for fast construction of MLX objects when data and dimensions are
  already known.
- Added
  [`mlx_fft()`](https://hughjonesd.github.io/Rmlx/reference/mlx_fft.md),
  [`mlx_fft2()`](https://hughjonesd.github.io/Rmlx/reference/mlx_fft.md),
  and
  [`mlx_fftn()`](https://hughjonesd.github.io/Rmlx/reference/mlx_fft.md)
  wrappers around MLX FFT kernels.
- Added distribution functions `mlx_d/p/qnorm()`, `mlx_d/p/qunif()` etc.
- Added
  [`mlx_quantile()`](https://hughjonesd.github.io/Rmlx/reference/mlx_quantile.md).
- Added
  [`mlx_coordinate_descent()`](https://hughjonesd.github.io/Rmlx/reference/mlx_coordinate_descent.md),
  a coordinate descent algorithm.
- Fixed several `[`/`[<-` bugs affecting non-contiguous, unsorted, and
  duplicate subsetting patterns on `mlx` arrays.
- [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md)
  now takes a much faster path for large numeric matrices by letting MLX
  handle column-major inputs directly.
- Base reducers [`all()`](https://rdrr.io/r/base/all.html) and
  [`any()`](https://rdrr.io/r/base/any.html) applied to mlx arrays now
  return plain R logical scalars;
  [`mlx_all()`](https://hughjonesd.github.io/Rmlx/reference/mlx_sum.md)/[`mlx_any()`](https://hughjonesd.github.io/Rmlx/reference/mlx_sum.md)
  continue to yield mlx booleans.
- Added mlx-aware wrappers for
  [`row()`](https://hughjonesd.github.io/Rmlx/reference/row.md),
  [`col()`](https://hughjonesd.github.io/Rmlx/reference/row.md),
  [`asplit()`](https://hughjonesd.github.io/Rmlx/reference/asplit.md),
  and
  [`backsolve()`](https://hughjonesd.github.io/Rmlx/reference/mlx_solve_triangular.md).
- Added
  [`scale.mlx()`](https://hughjonesd.github.io/Rmlx/reference/scale.mlx.md)
  to center/scale matrices entirely on the MLX backend (with MLX arrays
  stored in the `scaled:center` / `scaled:scale` attributes).
- [`scale.mlx()`](https://hughjonesd.github.io/Rmlx/reference/scale.mlx.md)
  now always records its `scaled:center` / `scaled:scale` attributes as
  1 x p MLX arrays, keeping them lazily evaluated even after coercion.
- [`as.matrix.mlx()`](https://hughjonesd.github.io/Rmlx/reference/as.matrix.mlx.md)
  now preserves any user-set attributes (including the MLX scaling
  metadata) when copying arrays back to base R.
- Created a new benchmarks vignette.
- Added pre-commit hooks to run, commit and print benchmark.

## Rmlx 0.1.0

- Initial release on r-universe.
