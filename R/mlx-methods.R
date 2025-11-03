#' Base R generics with mlx methods
#'
#' Rmlx provides S3 methods for a number of base R generics so that common
#' operations keep working after converting objects with [as_mlx()]. The main
#' entry points are:
#'
#' @details
#' - [`%*%`](%*%.mlx) for matrix multiplication
#' - [`[`]([.mlx) and [`[<-`]([<-.mlx) for extraction and assignment
#' - [`Ops`](Ops.mlx) and [`Math`](Math.mlx) for elementwise arithmetic and math
#' - [`Summary`](Summary.mlx) for reductions such as `sum()` and `max()`;
#'   also [`mean()`](mean.mlx), [`length()`](length.mlx) and [`all.equal()`](all.equal.mlx).
#' - [`diag()`](diag.mlx), [`dim()`](dim.mlx) and [`dim<-`](dim<-.mlx)
#' - [`as.matrix()`](as.matrix.mlx), [`as.array()`](as.array.mlx), and [`as.vector()`](as.vector.mlx) for conversion back to base R
#' - [`row()`](row) and [`col()`](col) for index helpers that play nicely with mlx arrays
#' - [`cbind()`](cbind.mlx) and [`rbind()`](rbind.mlx) for binding arrays along rows or columns;
#'   there is also an [abind()] function modelled on `abind::abind()`.
#' - [`rowMeans()`](rowMeans.mlx), [`colMeans()`](colMeans.mlx), [`rowSums()`](rowSums.mlx), and [`colSums()`](colSums.mlx) for axis-wise summaries
#' - [`aperm()`](aperm.mlx), [`t()`](t.mlx), and [`dim<-`](\`dim<-.mlx\`) for shape manipulation
#' - [`kronecker()`](kronecker), [`outer()`](outer.mlx), [`crossprod()`](crossprod), and [`tcrossprod()`](tcrossprod) for linear algebra helpers
#' - [`fft()`](fft.mlx), [`chol()`](chol.mlx), [`chol2inv()`](chol2inv), [`backsolve()`](backsolve), and [`solve()`](solve.mlx) for numerical routines
#' - [`asplit()`](asplit) to slice arrays along a margin while staying on the MLX backend
#' - [`is.finite()`](is.finite.mlx), [`is.infinite()`](is.infinite.mlx) and [`is.nan()`](is.nan.mlx)
#' @seealso [as_mlx()]
#'
#' @name mlx-methods
#' @keywords documentation
NULL
