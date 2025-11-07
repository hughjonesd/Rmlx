# Changelog

## Rmlx 0.1.0.9000 (development)

- Added negative numeric indexing support for `[`/`[<-` on `mlx` arrays
  and documented subsetting semantics.
- Added
  [`mlx_fft()`](https://hughjonesd.github.io/Rmlx/reference/mlx_fft.md),
  [`mlx_fft2()`](https://hughjonesd.github.io/Rmlx/reference/mlx_fft.md),
  and
  [`mlx_fftn()`](https://hughjonesd.github.io/Rmlx/reference/mlx_fft.md)
  wrappers around MLX FFT kernels and aligned pkgdown coverage.
- Fixed several `[`/`[<-` bugs affecting non-contiguous, unsorted, and
  duplicate subsetting patterns on `mlx` arrays.
- [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md)
  now takes a much faster path for large numeric matrices by letting MLX
  handle column-major inputs directly; 20-iteration GPU benchmarks show
  6×–10× speedups for 1000–4000 square matrices and similar gains on
  CPU.
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

## Rmlx 0.1.0

- Initial release on r-universe.
