#' Row-bind mlx arrays
#'
#' @param ... Objects to bind. mlx arrays are kept in MLX; other inputs are
#'   coerced via `as_mlx()`.
#' @param deparse.level Compatibility argument accepted for S3 dispatch; ignored.
#' @return An mlx array stacked along the first axis.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.concatenate}
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' y <- as_mlx(matrix(5:8, 2, 2))
#' rbind(x, y)
rbind.mlx <- function(..., deparse.level = 1) {
  objs <- list(...)
  if (!length(objs)) stop("No objects to bind.", call. = FALSE)
  mlx_objs <- lapply(objs, function(x) if (is.mlx(x)) x else as_mlx(x))
  ptr <- cpp_mlx_concat(mlx_objs, 0L)
  ref <- mlx_objs[[1]]
  new_dim <- as.integer(c(sum(vapply(mlx_objs, function(t) t$dim[1], integer(1))), ref$dim[-1]))
  new_mlx(ptr, new_dim, ref$dtype, ref$device)
}

#' Column-bind mlx arrays
#'
#' @inheritParams rbind.mlx
#' @return An mlx array stacked along the second axis.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.concatenate}
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' y <- as_mlx(matrix(5:8, 2, 2))
#' cbind(x, y)
cbind.mlx <- function(..., deparse.level = 1) {
  objs <- list(...)
  if (!length(objs)) stop("No objects to bind.", call. = FALSE)
  mlx_objs <- lapply(objs, function(x) if (is.mlx(x)) x else as_mlx(x))
  ptr <- cpp_mlx_concat(mlx_objs, 1L)
  ref <- mlx_objs[[1]]
  new_dim <- as.integer(c(ref$dim[1], sum(vapply(mlx_objs, function(t) t$dim[2], integer(1)))))
  new_mlx(ptr, new_dim, ref$dtype, ref$device)
}
