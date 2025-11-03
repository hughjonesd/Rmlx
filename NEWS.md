

# Rmlx 0.1.0.9000 (development)

* Added negative numeric indexing support for `[`/`[<-` on `mlx` arrays and
  documented subsetting semantics.
* Added `mlx_fft()`, `mlx_fft2()`, and `mlx_fftn()` wrappers around MLX FFT kernels and aligned pkgdown coverage.
* Fixed several `[`/`[<-` bugs affecting non-contiguous, unsorted, and duplicate
  subsetting patterns on `mlx` arrays.
* `as_mlx()` now takes a much faster path for large numeric matrices by letting
  MLX handle column-major inputs directly; 20-iteration GPU benchmarks show
  6×–10× speedups for 1000–4000 square matrices and similar gains on CPU.
* Base reducers [all()] and [any()] applied to mlx arrays now return plain R
  logical scalars; `mlx_all()`/`mlx_any()` continue to yield mlx booleans.
* Added mlx-aware wrappers for [row()], [col()], [asplit()], and [backsolve()] so these helpers work without pulling data back to base R.

# Rmlx 0.1.0

* Initial release on r-universe.
