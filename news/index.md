# Changelog

## Rmlx 0.4.0

- [`scale()`](https://rdrr.io/r/base/scale.html) now preserves
  MLX-supplied center and scale vectors as MLX attributes, avoiding a
  round trip through base R.
- `mlx` objects can now have row, column and dimnames, and vectors can
  have names. These work much like base R. Names for any dimension may
  be `NULL`. Subsetting with a non-existent name is an error. Many
  functions, e.g. [`solve()`](https://rdrr.io/r/base/solve.html), `%*%`,
  etc. preserve dimnames like their R analogues.

## Rmlx 0.3.0

- We now allow float64 arrays. (Note that mlx doesn’t yet do operations
  on the gpu for these, however.)
- Breaking change: arrays no longer have an associated device. Instead,
  use
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md),
  [`with_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md)
  or
  [`local_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md)
  to choose a device to work on. This also makes irrelevant a bug in
  which many/most operations were silently ignoring their operands’
  `device`.
- Operations which are currently cpu-only used to switch silently to
  cpu. They now take a `device` argument which must be explicitly set to
  `"cpu"`. This is a teaching tool to help the user understand what
  operations can take place on which device.
- [`rbind()`](https://rdrr.io/r/base/cbind.html) and
  [`cbind()`](https://rdrr.io/r/base/cbind.html) now accept 1D vectors.
- Bugfix:
  [`mlx_expand_dims()`](https://hughjonesd.github.io/Rmlx/reference/mlx_expand_dims.md)
  works with 1D vectors.

## Rmlx 0.2.3

- Speeded up single-axis `[` subsetting by routing it through MLX
  `take()` instead of the generic meshgrid gather path.
- Added
  [`mlx_take_along_axis()`](https://hughjonesd.github.io/Rmlx/reference/mlx_take_along_axis.md),
  [`mlx_put_along_axis()`](https://hughjonesd.github.io/Rmlx/reference/mlx_put_along_axis.md),
  and
  [`mlx_scatter_add_axis()`](https://hughjonesd.github.io/Rmlx/reference/mlx_scatter_add_axis.md)
  wrappers for MLX’s axis-aligned indexed update ops.
- Added
  [`mlx_metal_kernel()`](https://hughjonesd.github.io/Rmlx/reference/mlx_metal_kernel.md)
  to build custom Metal kernels from R and keep inputs and outputs as
  MLX arrays.
- Updated the bundled MLX source tarball and minimum supported MLX
  version to 0.31.1.
- Fixed the Linux bundled-build LAPACK shim for complex
  eigendecomposition and SVD entry points used by MLX 0.31.1.
- Fixed compatibility with MLX 0.31.1.
- Force clean recompilation when the detected MLX install or build flags
  change, avoiding stale object files after MLX upgrades.

## Rmlx 0.2.2

## Rmlx 0.2.1

## Rmlx 0.2.0

## Rmlx 0.1.0.9000 (development)

- Speeded up subset assignment. We also now fail on NAs in indices.
- Exposed
  [`mlx_cast()`](https://hughjonesd.github.io/Rmlx/reference/mlx_cast.md)
  for casting arrays between dtypes/devices (previously internal
  `.mlx_cast`).
- `%*%` now requires both its arguments to be matrices (unlike base R).
- `as_mlx(x)` no longer returns scalars if `x` is a length-one vector.
- `mlx_arange(start, stop, step)` now matches
  [`seq()`](https://rdrr.io/r/base/seq.html) behavior (stop included if
  reachable).
- New
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  to return device associated with `x`.
- Renamed `mlx_get_device()` to
  [`mlx_best_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_best_device.md).
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
  [`mlx_gather_qmm()`](https://hughjonesd.github.io/Rmlx/reference/mlx_gather_qmm.md),
  and
  [`mlx_embedding()`](https://hughjonesd.github.io/Rmlx/reference/mlx_embedding.md)
  now accept 1-based indices for inputs/outputs, consistently with R.
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
- [`mlx_array()`](https://hughjonesd.github.io/Rmlx/reference/mlx_array.md)/[`mlx_matrix()`](https://hughjonesd.github.io/Rmlx/reference/mlx_matrix.md)
  now recycle shorter payloads when they evenly divide the target shape.
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
