#' Base R generics with mlx methods
#'
#' Rmlx provides S3 methods for a number of base R generics so that common
#' operations keep working after converting objects with [as_mlx()]. The main
#' entry points are:
#'
#' @details
#' - `\%*\%` for matrix multiplication (see `?\%*%.mlx`)
#' - `[` and `[<-` for extraction and assignment (see `?[.mlx` and `?[<-.mlx`)
#' - `Ops` and `Math` for elementwise arithmetic and math (see `?Ops.mlx` and `?Math.mlx`)
#' - `Summary` for reductions such as `sum()` and `max()` (see `?Summary.mlx`)
#' - `as.matrix()`, `as.array()`, and `as.vector()` for conversion back to base R (see `?as.matrix.mlx`, `?as.array.mlx`, `?as.vector.mlx`)
#' - `cbind()` and `rbind()` for binding arrays along rows or columns (see `?cbind.mlx` and `?rbind.mlx`)
#' - `rowMeans()`, `colMeans()`, `rowSums()`, and `colSums()` for axis-wise summaries (see `?rowMeans.mlx`, `?colMeans.mlx`, `?rowSums.mlx`, `?colSums.mlx`)
#' - `aperm()`, `t()`, and `dim<-` for shape manipulation (see `?aperm.mlx`, `?t.mlx`, `?\`dim<-.mlx\``)
#' - `kronecker()`, `outer()`, `crossprod()`, and `tcrossprod()` for linear algebra helpers (see `?kronecker`, `?outer.mlx`, `?crossprod`, `?tcrossprod`)
#' - `fft()`, `chol()`, `chol2inv()`, and `solve()` for numerical routines (see `?fft`, `?chol.mlx`, `?chol2inv`, `?solve.mlx`)
#'
#' @seealso [as_mlx()]
#'
#' @name base-generics
#' @keywords documentation
NULL
