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

#' Reduce MLX tensors
#'
#' These helpers mirror NumPy-style reductions, optionally collapsing one or
#' more axes. Use `drop = FALSE` to retain reduced axes with length one
#' (akin to `keepdims = TRUE` in NumPy).
#'
#' @param x An object coercible to `mlx` via [as_mlx()].
#' @param axis Optional integer vector of axes (1-indexed) to reduce.
#'   When `NULL`, the reduction is performed over all elements.
#' @param drop Logical flag controlling dimension dropping: `TRUE` (default)
#'   removes reduced axes, while `FALSE` retains them with length one.
#' @return An `mlx` tensor containing the reduction result.
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_sum(x)
#' mlx_sum(x, axis = 1)
#' mlx_prod(x, axis = 2, drop = FALSE)
#' mlx_all(x > 0)
#' mlx_any(x > 3)
#' @name mlx_sum
NULL

#' @rdname mlx_sum
#' @export
mlx_sum <- function(x, axis = NULL, drop = TRUE) {
  .mlx_reduce_dispatch(x, "sum", axis = axis, drop = drop)
}

#' @rdname mlx_sum
#' @export
mlx_prod <- function(x, axis = NULL, drop = TRUE) {
  .mlx_reduce_dispatch(x, "prod", axis = axis, drop = drop)
}

#' @rdname mlx_sum
#' @export
mlx_all <- function(x, axis = NULL, drop = TRUE) {
  .mlx_reduce_dispatch(x, "all", axis = axis, drop = drop)
}

#' @rdname mlx_sum
#' @export
mlx_any <- function(x, axis = NULL, drop = TRUE) {
  .mlx_reduce_dispatch(x, "any", axis = axis, drop = drop)
}

#' Mean of MLX array elements
#'
#' @param x An `mlx` object
#' @param ... Additional arguments (ignored)
#' @return An `mlx` scalar
#' @export
#' @method mean mlx
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mean(x)
mean.mlx <- function(x, ...) {
  .mlx_reduce(x, "mean")
}

#' Row means for MLX tensors
#'
#' @inheritParams mlx_reduction_base
#' @return An `mlx` tensor if `x` is `mlx`, otherwise a numeric vector.
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 3, 2))
#' rowMeans(x)
rowMeans <- function(x, na.rm = FALSE, dims = 1, ...) {
  if (inherits(x, "mlx")) {
    return(.mlx_reduce_axis(x, "mean", axis = 2L, keepdims = FALSE))
  }
  base::rowMeans(x, na.rm = na.rm, dims = dims, ...)
}

#' Column means for MLX tensors
#'
#' @inheritParams mlx_reduction_base
#' @return An `mlx` tensor if `x` is `mlx`, otherwise a numeric vector.
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 3, 2))
#' colMeans(x)
colMeans <- function(x, na.rm = FALSE, dims = 1, ...) {
  if (inherits(x, "mlx")) {
    return(.mlx_reduce_axis(x, "mean", axis = 1L, keepdims = FALSE))
  }
  base::colMeans(x, na.rm = na.rm, dims = dims, ...)
}

#' Row sums for MLX tensors
#'
#' @inheritParams mlx_reduction_base
#' @return An `mlx` tensor if `x` is `mlx`, otherwise a numeric vector.
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 3, 2))
#' rowSums(x)
rowSums <- function(x, na.rm = FALSE, dims = 1, ...) {
  if (inherits(x, "mlx")) {
    return(.mlx_reduce_axis(x, "sum", axis = 2L, keepdims = FALSE))
  }
  base::rowSums(x, na.rm = na.rm, dims = dims, ...)
}

#' Column sums for MLX tensors
#'
#' @inheritParams mlx_reduction_base
#' @return An `mlx` tensor if `x` is `mlx`, otherwise a numeric vector.
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 3, 2))
#' colSums(x)
colSums <- function(x, na.rm = FALSE, dims = 1, ...) {
  if (inherits(x, "mlx")) {
    return(.mlx_reduce_axis(x, "sum", axis = 1L, keepdims = FALSE))
  }
  base::colSums(x, na.rm = na.rm, dims = dims, ...)
}

#' Transpose of MLX matrix
#'
#' @param x An `mlx` matrix
#' @return Transposed `mlx` matrix
#' @export
#' @method t mlx
#' @examples
#' x <- as_mlx(matrix(1:6, 2, 3))
#' t(x)
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
#' @examples
#' x <- as_mlx(matrix(1:6, 2, 3))
#' crossprod(x)
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
#' @examples
#' x <- as_mlx(matrix(1:6, 2, 3))
#' tcrossprod(x)
tcrossprod.mlx <- function(x, y = NULL, ...) {
  if (is.null(y)) y <- x
  x %*% t(y)
}

#' Reduce an MLX tensor over all axes
#'
#' @param x An `mlx` tensor.
#' @param op Reduction identifier.
#' @return Reduced `mlx` tensor.
#' @noRd
.mlx_reduce <- function(x, op) {
  ptr <- cpp_mlx_reduce(x$ptr, op)
  .mlx_wrap_result(ptr, x$device)
}

#' Reduce an MLX tensor along a specific axis
#'
#' @param x An `mlx` tensor.
#' @param op Reduction identifier.
#' @param axis Axis index (zero-based).
#' @param keepdims Logical flag preserving reduced axis.
#' @return Reduced `mlx` tensor.
#' @noRd
.mlx_reduce_axis <- function(x, op, axis, keepdims) {
  axis0 <- as.integer(axis) - 1L
  if (axis0 < 0L || axis0 >= length(x$dim)) {
    stop("axis is out of bounds for input tensor", call. = FALSE)
  }
  ptr <- cpp_mlx_reduce_axis(x$ptr, op, axis0, keepdims)
  .mlx_wrap_result(ptr, x$device)
}

.mlx_reduce_axes <- function(x, op, axes, drop) {
  axes <- as.integer(axes)
  if (any(is.na(axes))) {
    stop("axis must be a vector of integers", call. = FALSE)
  }
  ndim <- length(x$dim)
  if (any(axes < 1L | axes > ndim)) {
    stop("axis is out of bounds for input tensor", call. = FALSE)
  }
  axes <- unique(axes)
  if (!drop) {
    for (ax in sort(axes)) {
      x <- .mlx_reduce_axis(x, op, axis = ax, keepdims = TRUE)
    }
  } else {
    for (ax in sort(axes, decreasing = TRUE)) {
      x <- .mlx_reduce_axis(x, op, axis = ax, keepdims = FALSE)
    }
  }
  x
}

.mlx_reduce_dispatch <- function(x, op, axis = NULL, drop = TRUE) {
  x <- if (inherits(x, "mlx")) x else as_mlx(x)
  if (is.null(axis)) {
    return(.mlx_reduce(x, op))
  }
  if (!is.logical(drop) || length(drop) != 1L) {
    stop("drop must be a single logical value", call. = FALSE)
  }
  axes <- axis
  .mlx_reduce_axes(x, op, axes, drop = drop)
}

#' @export
Summary.mlx <- function(x, ..., na.rm = FALSE) {
  op <- .Generic
  if (!(op %in% c("sum", "prod", "all", "any"))) {
    stop("Operation not implemented for mlx objects: ", op, call. = FALSE)
  }
  if (na.rm) {
    warning("na.rm is ignored for mlx tensors", call. = FALSE)
  }

  args <- list(x, ...)
  reduce_one <- function(obj) {
    obj_mlx <- if (inherits(obj, "mlx")) obj else as_mlx(obj)
    .mlx_reduce(obj_mlx, switch(op,
      sum = "sum",
      prod = "prod",
      all = "all",
      any = "any"
    ))
  }

  result <- reduce_one(args[[1L]])
  if (length(args) > 1L) {
    for (obj in args[-1L]) {
      scalar <- reduce_one(obj)
      result <- switch(op,
        sum = result + scalar,
        prod = result * scalar,
        all = result & scalar,
        any = result | scalar
      )
    }
  }
  result
}
