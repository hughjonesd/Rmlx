# Base R generics with mlx methods

Rmlx provides S3 methods for a number of base R generics so that common
operations keep working after converting objects with
[`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md). The
main entry points are:

## Details

- [`%*%`](NA) for matrix multiplication

- [`[`](https://hughjonesd.github.io/Rmlx/reference/%5B.mlx) and
  [`[<-`](NA) for extraction and assignment

- [`Ops`](https://hughjonesd.github.io/Rmlx/reference/Ops.mlx) and
  [`Math`](https://hughjonesd.github.io/Rmlx/reference/Math.mlx) for
  elementwise arithmetic and math

- [`Summary`](https://hughjonesd.github.io/Rmlx/reference/Summary.mlx)
  for reductions such as [`sum()`](https://rdrr.io/r/base/sum.html) and
  [`max()`](https://rdrr.io/r/base/Extremes.html); also
  [[`mean()`](https://rdrr.io/r/base/mean.html)](https://hughjonesd.github.io/Rmlx/reference/mean.mlx),
  [[`length()`](https://rdrr.io/r/base/length.html)](https://hughjonesd.github.io/Rmlx/reference/length.mlx)
  and
  [[`all.equal()`](https://rdrr.io/r/base/all.equal.html)](https://hughjonesd.github.io/Rmlx/reference/all.equal.mlx).

- [[`diag()`](https://hughjonesd.github.io/Rmlx/reference/diag.md)](https://hughjonesd.github.io/Rmlx/reference/diag.mlx),
  [[`dim()`](https://rdrr.io/r/base/dim.html)](https://hughjonesd.github.io/Rmlx/reference/dim.mlx)
  and [`dim<-`](NA)

- [[`as.matrix()`](https://rdrr.io/r/base/matrix.html)](https://hughjonesd.github.io/Rmlx/reference/as.matrix.mlx),
  [[`as.array()`](https://rdrr.io/r/base/array.html)](https://hughjonesd.github.io/Rmlx/reference/as.array.mlx),
  and
  [[`as.vector()`](https://rdrr.io/r/base/vector.html)](https://hughjonesd.github.io/Rmlx/reference/as.vector.mlx)
  for conversion back to base R

- [[`row()`](https://hughjonesd.github.io/Rmlx/reference/row.md)](https://hughjonesd.github.io/Rmlx/reference/row)
  and
  [[`col()`](https://hughjonesd.github.io/Rmlx/reference/row.md)](https://hughjonesd.github.io/Rmlx/reference/col)
  for index helpers that play nicely with mlx arrays

- [[`cbind()`](https://rdrr.io/r/base/cbind.html)](https://hughjonesd.github.io/Rmlx/reference/cbind.mlx)
  and
  [[`rbind()`](https://rdrr.io/r/base/cbind.html)](https://hughjonesd.github.io/Rmlx/reference/rbind.mlx)
  for binding arrays along rows or columns; there is also an
  [`abind()`](https://hughjonesd.github.io/Rmlx/reference/abind.md)
  function modelled on `abind::abind()`.

- [[`rowMeans()`](https://hughjonesd.github.io/Rmlx/reference/rowMeans.md)](https://hughjonesd.github.io/Rmlx/reference/rowMeans.mlx),
  [[`colMeans()`](https://hughjonesd.github.io/Rmlx/reference/colMeans.md)](https://hughjonesd.github.io/Rmlx/reference/colMeans.mlx),
  [[`rowSums()`](https://hughjonesd.github.io/Rmlx/reference/rowSums.md)](https://hughjonesd.github.io/Rmlx/reference/rowSums.mlx),
  and
  [[`colSums()`](https://hughjonesd.github.io/Rmlx/reference/colSums.md)](https://hughjonesd.github.io/Rmlx/reference/colSums.mlx)
  for axis-wise summaries

- [[`aperm()`](https://rdrr.io/r/base/aperm.html)](https://hughjonesd.github.io/Rmlx/reference/aperm.mlx),
  [[`t()`](https://rdrr.io/r/base/t.html)](https://hughjonesd.github.io/Rmlx/reference/t.mlx),
  and [`dim<-`](NA) for shape manipulation

- [[`kronecker()`](https://hughjonesd.github.io/Rmlx/reference/kronecker.md)](https://hughjonesd.github.io/Rmlx/reference/kronecker),
  [[`outer()`](https://hughjonesd.github.io/Rmlx/reference/outer.md)](https://hughjonesd.github.io/Rmlx/reference/outer.mlx),
  [[`crossprod()`](https://rdrr.io/r/base/crossprod.html)](https://hughjonesd.github.io/Rmlx/reference/crossprod),
  and
  [[`tcrossprod()`](https://rdrr.io/r/base/crossprod.html)](https://hughjonesd.github.io/Rmlx/reference/tcrossprod)
  for linear algebra helpers

- [[`fft()`](https://hughjonesd.github.io/Rmlx/reference/fft.md)](https://hughjonesd.github.io/Rmlx/reference/fft.mlx),
  [[`chol()`](https://rdrr.io/r/base/chol.html)](https://hughjonesd.github.io/Rmlx/reference/chol.mlx),
  [[`chol2inv()`](https://hughjonesd.github.io/Rmlx/reference/chol2inv.md)](https://hughjonesd.github.io/Rmlx/reference/chol2inv),
  [[`backsolve()`](https://hughjonesd.github.io/Rmlx/reference/mlx_solve_triangular.md)](https://hughjonesd.github.io/Rmlx/reference/backsolve),
  and
  [[`solve()`](https://rdrr.io/r/base/solve.html)](https://hughjonesd.github.io/Rmlx/reference/solve.mlx)
  for numerical routines

- [[`scale()`](https://rdrr.io/r/base/scale.html)](https://hughjonesd.github.io/Rmlx/reference/scale.mlx)
  for column-wise centring and scaling that stays on the MLX backend

- [[`asplit()`](https://hughjonesd.github.io/Rmlx/reference/asplit.md)](https://hughjonesd.github.io/Rmlx/reference/asplit)
  to slice arrays along a margin while staying on the MLX backend

- [[`is.finite()`](https://rdrr.io/r/base/is.finite.html)](https://hughjonesd.github.io/Rmlx/reference/is.finite.mlx),
  [[`is.infinite()`](https://rdrr.io/r/base/is.finite.html)](https://hughjonesd.github.io/Rmlx/reference/is.infinite.mlx)
  and
  [[`is.nan()`](https://rdrr.io/r/base/is.finite.html)](https://hughjonesd.github.io/Rmlx/reference/is.nan.mlx)

Most methods return mlx objects. One exception is that
[`all()`](https://rdrr.io/r/base/all.html) and
[`any()`](https://rdrr.io/r/base/any.html) return standard R `TRUE` or
`FALSE` when used on mlx objects.

## See also

[`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md)
