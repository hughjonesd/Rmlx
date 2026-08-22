# GPU QR reduction for tall least-squares problems

Computes the QR quantities needed by large least-squares fits without
materializing the full `Q` matrix. This is intended for tall, full-rank
real-valued matrices. It returns the final upper-triangular `R` and,
when `y` is supplied, `qty = Q' y`.

## Usage

``` r
mlx_qr_gpu(
  x,
  y = NULL,
  block_rows = NULL,
  tol = 1e-04,
  method = c("cholqr", "tsqr")
)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- y:

  Optional response vector or matrix with `nrow(x)` rows.

- block_rows:

  Number of rows reduced by each first-level GPU block for
  `method = "tsqr"`. The default `NULL` chooses a GPU tile size
  automatically.

- tol:

  Relative tolerance for detecting rank deficiency from `diag(R)`.

- method:

  `"cholqr"` for two Cholesky QR passes or `"tsqr"` for the custom Metal
  tall-skinny QR reduction.

## Value

A list with components `R`, optional `qty`, `rank`, `pivot`, and
`block_rows`. Well-conditioned Cholesky QR fits with `y` also return
`qty_corrected` for the coefficient solve.

## Details

The default `"cholqr"` method applies two Cholesky QR passes, with the
large matrix products on the GPU. The `"tsqr"` method uses custom Metal
kernels for a tiled Householder reduction followed by a tree reduction
of the small triangular factors.

GPU work is currently restricted to `float32`. Integer inputs are cast
to `float32`; `float64` and complex inputs are not supported on the GPU
path.

`method = "tsqr"` stores one input tile in Metal threadgroup memory.
When `block_rows = NULL`, the tile height is chosen from `p` and
`ncol(y)` to fit the 32 KB threadgroup-memory limit and provide enough
independent blocks to occupy the GPU.

Cholesky QR checks the orthogonality of its first pass. If that pass is
unsafe, it falls back to GPU TSQR when its compact state fits in
threadgroup memory, and otherwise to MLX QR on the CPU. For
well-conditioned fits with `y`, an MLX GPU residual-correction pass
returns `qty_corrected` for a more accurate least-squares solve while
preserving `qty = Q' y`.

## Examples

``` r
if (FALSE) { # \dontrun{
x <- as_mlx(matrix(rnorm(1000 * 8), 1000, 8))
y <- as_mlx(matrix(rnorm(1000), 1000, 1))
fit <- mlx_qr_gpu(x, y)
coef <- mlx_solve_triangular(fit$R, fit$qty, upper = TRUE, device = "cpu")
} # }
```
