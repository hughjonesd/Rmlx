#' Row-bind mlx arrays
#'
#' @param ... Objects to bind. mlx arrays are kept in MLX; other inputs are
#'   coerced via `as_mlx()`.
#' @param deparse.level Compatibility argument accepted for S3 dispatch; ignored.
#' @return An mlx array stacked along the first axis.
#' @details Unlike base R's `rbind()`, this function supports arrays with more
#'   than 2 dimensions and preserves all dimensions except the first (which is
#'   summed across inputs). Base R's `rbind()` flattens higher-dimensional arrays
#'   to matrices before binding.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.concatenate}
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' y <- as_mlx(matrix(5:8, 2, 2))
#' rbind(x, y)
rbind.mlx <- function(..., deparse.level = 1) {
  objs <- list(...)
  if (!length(objs)) stop("No objects to bind.", call. = FALSE)
  mlx_objs <- lapply(objs, as_mlx)
  ptr <- cpp_mlx_concat(mlx_objs, 0L)
  ref <- mlx_objs[[1]]
  dim1s <- vapply(mlx_objs, function(t) t$dim[1], integer(1))
  new_dim1 <- sum(dim1s)
  new_dim <- as.integer(c(new_dim1, ref$dim[-1]))
  new_mlx(ptr, new_dim, ref$dtype, ref$device)
}

#' Column-bind mlx arrays
#'
#' @inheritParams rbind.mlx
#' @return An mlx array stacked along the second axis.
#' @details Unlike base R's `cbind()`, this function supports arrays with more
#'   than 2 dimensions and preserves all dimensions except the second (which is
#'   summed across inputs). Base R's `cbind()` flattens higher-dimensional arrays
#'   to matrices before binding.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.concatenate}
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' y <- as_mlx(matrix(5:8, 2, 2))
#' cbind(x, y)
cbind.mlx <- function(..., deparse.level = 1) {
  objs <- list(...)
  if (!length(objs)) stop("No objects to bind.", call. = FALSE)
  mlx_objs <- lapply(objs, as_mlx)
  ptr <- cpp_mlx_concat(mlx_objs, 1L)
  ref <- mlx_objs[[1]]
  dim2s <- vapply(mlx_objs, function(t) t$dim[2], integer(1))
  new_dim2 <- sum(dim2s)
  new_dim <- as.integer(c(ref$dim[1], new_dim2, ref$dim[-(1:2)]))
  new_mlx(ptr, new_dim, ref$dtype, ref$device)
}
