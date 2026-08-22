# Base R generics with mlx methods

Rmlx provides S3 methods for a number of base R generics so that common
operations keep working after converting objects with
[`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md). The
main entry points are:

## Details

- [`%*%`](NA) for matrix multiplication

- [`[`](NA) and [`[<-`](NA) for extraction and assignment

- [`Ops`](https://hughjonesd.github.io/Rmlx/reference/Ops.mlx) and
  [`Math`](https://hughjonesd.github.io/Rmlx/reference/Math.mlx) for
  elementwise arithmetic and math

- [`Summary`](https://hughjonesd.github.io/Rmlx/reference/Summary.mlx)
  for reductions such as [`sum()`](https://rdrr.io/r/base/sum.html) and
  [`max()`](https://rdrr.io/r/base/Extremes.html); also
  [`mean()`](https://hughjonesd.github.io/Rmlx/reference/mean.mlx),
  [`length()`](https://hughjonesd.github.io/Rmlx/reference/length.mlx)
  and
  [`all.equal()`](https://hughjonesd.github.io/Rmlx/reference/all.equal.mlx).

- [`diag()`](https://hughjonesd.github.io/Rmlx/reference/diag.mlx),
  [`dim()`](https://hughjonesd.github.io/Rmlx/reference/dim.mlx) and
  [`dim<-`](NA)

- [`as_r()`](https://hughjonesd.github.io/Rmlx/reference/as_r),
  [`as.matrix()`](https://hughjonesd.github.io/Rmlx/reference/as.matrix.mlx),
  [`as.array()`](https://hughjonesd.github.io/Rmlx/reference/as.array.mlx),
  and
  [`as.vector()`](https://hughjonesd.github.io/Rmlx/reference/as.vector.mlx)
  for conversion back to base R

- [`row()`](https://hughjonesd.github.io/Rmlx/reference/row) and
  [`col()`](https://hughjonesd.github.io/Rmlx/reference/col) for index
  helpers that play nicely with mlx arrays

- [`cbind()`](https://hughjonesd.github.io/Rmlx/reference/cbind.mlx) and
  [`rbind()`](https://hughjonesd.github.io/Rmlx/reference/rbind.mlx) for
  binding arrays along rows or columns; there is also an
  [`abind()`](https://hughjonesd.github.io/Rmlx/reference/abind.md)
  function modelled on `abind::abind()`.

- [`rowMeans()`](https://hughjonesd.github.io/Rmlx/reference/rowMeans.mlx),
  [`colMeans()`](https://hughjonesd.github.io/Rmlx/reference/colMeans.mlx),
  [`rowSums()`](https://hughjonesd.github.io/Rmlx/reference/rowSums.mlx),
  and
  [`colSums()`](https://hughjonesd.github.io/Rmlx/reference/colSums.mlx)
  for axis-wise summaries

- [`aperm()`](https://hughjonesd.github.io/Rmlx/reference/aperm.mlx),
  [`t()`](https://hughjonesd.github.io/Rmlx/reference/t.mlx), and
  [`dim<-`](NA) for shape manipulation

- [`kronecker()`](https://hughjonesd.github.io/Rmlx/reference/kronecker),
  [`outer()`](https://hughjonesd.github.io/Rmlx/reference/outer.mlx),
  [`crossprod()`](https://hughjonesd.github.io/Rmlx/reference/crossprod),
  and
  [`tcrossprod()`](https://hughjonesd.github.io/Rmlx/reference/tcrossprod)
  for linear algebra helpers

- [`fft()`](https://hughjonesd.github.io/Rmlx/reference/fft.mlx),
  [`chol()`](https://hughjonesd.github.io/Rmlx/reference/chol.mlx),
  [`chol2inv()`](https://hughjonesd.github.io/Rmlx/reference/chol2inv),
  [`backsolve()`](https://hughjonesd.github.io/Rmlx/reference/backsolve),
  and [`solve()`](https://hughjonesd.github.io/Rmlx/reference/solve.mlx)
  for numerical routines

- [`scale()`](https://hughjonesd.github.io/Rmlx/reference/scale.mlx) for
  column-wise centring and scaling that stays on the MLX backend

- [`asplit()`](https://hughjonesd.github.io/Rmlx/reference/asplit) to
  slice arrays along a margin while staying on the MLX backend

- [`is.finite()`](https://hughjonesd.github.io/Rmlx/reference/is.finite.mlx),
  [`is.infinite()`](https://hughjonesd.github.io/Rmlx/reference/is.infinite.mlx)
  and
  [`is.nan()`](https://hughjonesd.github.io/Rmlx/reference/is.nan.mlx)

Most methods return mlx objects. One exception is that
[`all()`](https://rdrr.io/r/base/all.html) and
[`any()`](https://rdrr.io/r/base/any.html) return standard R `TRUE` or
`FALSE` when used on mlx objects.

## See also

[`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md)
