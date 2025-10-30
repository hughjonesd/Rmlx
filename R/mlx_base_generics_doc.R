#' Base R generics with mlx methods
#'
#' Rmlx provides S3 methods for a number of base R generics so that common
#' operations keep working after converting objects with [as_mlx()]. The main
#' entry points are:
#'
#' @details
#' - [`%*%`][=%*%.mlx] for matrix multiplication
#' - [`[`][=[.mlx] and [`[<-`][=[<-.mlx] for extraction and assignment
#' - [Ops][=Ops.mlx] and [Math][=Math.mlx] for elementwise arithmetic and math
#' - [Summary][=Summary.mlx] for reductions such as `sum()` and `max()`
#' - [as.matrix()][=as.matrix.mlx], [as.array()][=as.array.mlx], and
#'   [as.vector()][=as.vector.mlx] for conversion back to base R
#' - [cbind()][=cbind.mlx] and [rbind()][=rbind.mlx] for binding arrays along
#'   rows or columns
#' - [aperm()][=aperm.mlx], [t()][=t.mlx], and [`dim<-`][=dim<-.mlx] for shape
#'   manipulation
#' - [kronecker()][=kronecker], [outer()][=outer.mlx], [crossprod()][=crossprod],
#'   and [tcrossprod()][=tcrossprod] for linear algebra helpers
#' - [fft()][=fft.mlx], [chol()][=chol.mlx], [chol2inv()][=chol2inv], and
#'   [solve()][=solve.mlx] for numerical routines
#'
#' See the individual help pages for usage examples and any mlx-specific notes.
#'
#' @name mlx_base_generics
#' @keywords documentation
NULL
