#' Shared arguments for MLX/base reduction helpers.
#'
#' @param x An array or `mlx` tensor.
#' @param na.rm Logical; currently ignored for `mlx` tensors.
#' @param dims Dimensions passed through to the base implementation when
#'   `x` is not an `mlx` tensor.
#' @param ... Additional arguments forwarded to the base implementation.
#' @keywords internal
#' @name mlx_reduction_base
NULL

#' Sum of MLX array elements
#'
#' @param x An `mlx` object
#' @param ... Additional arguments (ignored)
#' @param na.rm Ignored (for compatibility)
#' @return An `mlx` scalar
#' @export
#' @method sum mlx
sum.mlx <- function(x, ..., na.rm = FALSE) {
  .mlx_reduce(x, "sum")
}

#' Mean of MLX array elements
#'
#' @param x An `mlx` object
#' @param ... Additional arguments (ignored)
#' @return An `mlx` scalar
#' @export
#' @method mean mlx
mean.mlx <- function(x, ...) {
  .mlx_reduce(x, "mean")
}

#' Row means for MLX tensors
#'
#' @inheritParams mlx_reduction_base
#' @return An `mlx` tensor if `x` is `mlx`, otherwise a numeric vector.
#' @export
rowMeans <- function(x, na.rm = FALSE, dims = 1, ...) {
  if (inherits(x, "mlx")) {
    return(.mlx_reduce_axis(x, "mean", axis = 1L, keepdims = FALSE))
  }
  base::rowMeans(x, na.rm = na.rm, dims = dims, ...)
}

#' Column means for MLX tensors
#'
#' @inheritParams mlx_reduction_base
#' @return An `mlx` tensor if `x` is `mlx`, otherwise a numeric vector.
#' @export
colMeans <- function(x, na.rm = FALSE, dims = 1, ...) {
  if (inherits(x, "mlx")) {
    return(.mlx_reduce_axis(x, "mean", axis = 0L, keepdims = FALSE))
  }
  base::colMeans(x, na.rm = na.rm, dims = dims, ...)
}

#' Row sums for MLX tensors
#'
#' @inheritParams mlx_reduction_base
#' @return An `mlx` tensor if `x` is `mlx`, otherwise a numeric vector.
#' @export
rowSums <- function(x, na.rm = FALSE, dims = 1, ...) {
  if (inherits(x, "mlx")) {
    return(.mlx_reduce_axis(x, "sum", axis = 1L, keepdims = FALSE))
  }
  base::rowSums(x, na.rm = na.rm, dims = dims, ...)
}

#' Column sums for MLX tensors
#'
#' @inheritParams mlx_reduction_base
#' @return An `mlx` tensor if `x` is `mlx`, otherwise a numeric vector.
#' @export
colSums <- function(x, na.rm = FALSE, dims = 1, ...) {
  if (inherits(x, "mlx")) {
    return(.mlx_reduce_axis(x, "sum", axis = 0L, keepdims = FALSE))
  }
  base::colSums(x, na.rm = na.rm, dims = dims, ...)
}

#' Transpose of MLX matrix
#'
#' @param x An `mlx` matrix
#' @return Transposed `mlx` matrix
#' @export
#' @method t mlx
t.mlx <- function(x) {
  # Must transpose in MLX so MLX shape matches R dims
  # Layout conversion (physical reordering) happens at boundaries during copy
  ptr <- cpp_mlx_transpose(x$ptr)
  new_mlx(ptr, rev(x$dim), x$dtype, x$device)
}

#' Cross product
#'
#' @param x An `mlx` matrix
#' @param y An `mlx` matrix (default: NULL, uses x)
#' @return `t(x) %*% y` as an `mlx` object
#' @param ... Additional arguments passed to base::crossprod.
#' @export
#' @method crossprod mlx
crossprod.mlx <- function(x, y = NULL, ...) {
  if (is.null(y)) y <- x
  t(x) %*% y
}

#' Transposed cross product
#'
#' @param x An `mlx` matrix
#' @param y An `mlx` matrix (default: NULL, uses x)
#' @return `x %*% t(y)` as an `mlx` object
#' @param ... Additional arguments passed to base::tcrossprod.
#' @export
#' @method tcrossprod mlx
tcrossprod.mlx <- function(x, y = NULL, ...) {
  if (is.null(y)) y <- x
  x %*% t(y)
}

# Internal helper: full reduction
.mlx_reduce <- function(x, op) {
  ptr <- cpp_mlx_reduce(x$ptr, op)
  shape <- cpp_mlx_shape(ptr)
  if (length(shape) == 0) {
    new_dim <- integer(0)
  } else {
    new_dim <- as.integer(shape)
  }
  new_mlx(ptr, new_dim, x$dtype, x$device)
}

# Internal helper: axis reduction
.mlx_reduce_axis <- function(x, op, axis, keepdims) {
  ptr <- cpp_mlx_reduce_axis(x$ptr, op, as.integer(axis), keepdims)

  # Calculate new dimensions
  if (keepdims) {
    new_dim <- x$dim
    new_dim[axis + 1L] <- 1L  # R is 1-indexed
  } else {
    new_dim <- x$dim[-(axis + 1L)]  # Remove the reduced dimension
    if (length(new_dim) == 0) new_dim <- 1L
  }

  new_mlx(ptr, new_dim, x$dtype, x$device)
}
