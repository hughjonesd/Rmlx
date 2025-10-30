#' Shared arguments for MLX/base reduction helpers.
#'
#' @param x An array or mlx array.
#' @param na.rm Logical; currently ignored for mlx arrays.
#' @param dims Dimensions passed through to the base implementation when
#'   `x` is not an mlx array.
#' @param ... Additional arguments forwarded to the base implementation.
#' @keywords internal
#' @name mlx_reduction_base
NULL

#' Reduce mlx arrays
#'
#' These helpers mirror NumPy-style reductions, optionally collapsing one or
#' more axes. Use `drop = FALSE` to retain reduced axes with length one
#' (akin to setting `drop = FALSE` in base R).
#'
#' @inheritParams common_params
#' @param axis Optional integer vector of axes (1-indexed) to reduce.
#'   When `NULL`, the reduction is performed over all elements.
#' @param drop Logical flag controlling dimension dropping: `TRUE` (default)
#'   removes reduced axes, while `FALSE` retains them with length one.
#' @param ddof Non-negative integer delta degrees of freedom for variance or
#'   standard deviation reductions.
#' @return An mlx array containing the reduction result.
#' @seealso
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.sum},
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.prod},
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.all},
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.any},
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.mean},
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.var},
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.std}
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_sum(x)
#' mlx_sum(x, axis = 1)
#' mlx_prod(x, axis = 2, drop = FALSE)
#' mlx_all(x > 0)
#' mlx_any(x > 3)
#' mlx_mean(x, axis = 1)
#' mlx_var(x, axis = 2)
#' mlx_std(x, ddof = 1)
#' @aliases mlx_sum mlx_prod mlx_all mlx_any mlx_mean mlx_var mlx_std
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

#' @rdname mlx_sum
#' @export
mlx_mean <- function(x, axis = NULL, drop = TRUE) {
  .mlx_reduce_dispatch(x, "mean", axis = axis, drop = drop)
}

#' @rdname mlx_sum
#' @export
mlx_var <- function(x, axis = NULL, drop = TRUE, ddof = 0L) {
  .mlx_reduce_dispatch(x, "var", axis = axis, drop = drop, ddof = ddof)
}

#' @rdname mlx_sum
#' @export
mlx_std <- function(x, axis = NULL, drop = TRUE, ddof = 0L) {
  .mlx_reduce_dispatch(x, "std", axis = axis, drop = drop, ddof = ddof)
}

#' Mean of MLX array elements
#'
#' @inheritParams mlx_array_required
#' @param ... Additional arguments (ignored)
#' @return An mlx scalar
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.mean}
#' @export
#' @method mean mlx
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mean(x)
mean.mlx <- function(x, ...) {
  .mlx_reduce(x, "mean")
}

#' Row means for mlx arrays
#'
#' @inheritParams mlx_reduction_base
#' @return An mlx array if `x` is mlx, otherwise a numeric vector.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.mean}
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 3, 2))
#' rowMeans(x)
rowMeans <- function(x, na.rm = FALSE, dims = 1, ...) {
  if (inherits(x, "mlx")) {
    return(.mlx_reduce_axis(x, "mean", axis = 2L, drop = TRUE))
  }
  base::rowMeans(x, na.rm = na.rm, dims = dims, ...)
}

#' Column means for mlx arrays
#'
#' @inheritParams mlx_reduction_base
#' @return An mlx array if `x` is mlx, otherwise a numeric vector.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.mean}
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 3, 2))
#' colMeans(x)
colMeans <- function(x, na.rm = FALSE, dims = 1, ...) {
  if (inherits(x, "mlx")) {
    return(.mlx_reduce_axis(x, "mean", axis = 1L, drop = TRUE))
  }
  base::colMeans(x, na.rm = na.rm, dims = dims, ...)
}

#' Row sums for mlx arrays
#'
#' @inheritParams mlx_reduction_base
#' @return An mlx array if `x` is mlx, otherwise a numeric vector.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.sum}
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 3, 2))
#' rowSums(x)
rowSums <- function(x, na.rm = FALSE, dims = 1, ...) {
  if (inherits(x, "mlx")) {
    return(.mlx_reduce_axis(x, "sum", axis = 2L, drop = TRUE))
  }
  base::rowSums(x, na.rm = na.rm, dims = dims, ...)
}

#' Column sums for mlx arrays
#'
#' @inheritParams mlx_reduction_base
#' @return An mlx array if `x` is mlx, otherwise a numeric vector.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.sum}
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 3, 2))
#' colSums(x)
colSums <- function(x, na.rm = FALSE, dims = 1, ...) {
  if (inherits(x, "mlx")) {
    return(.mlx_reduce_axis(x, "sum", axis = 1L, drop = TRUE))
  }
  base::colSums(x, na.rm = na.rm, dims = dims, ...)
}

#' Transpose of MLX matrix
#'
#' @inheritParams mlx_matrix_required
#' @return Transposed mlx matrix
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.transpose}
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
#' @inheritParams mlx_matrix_required
#' @param y An mlx matrix (default: NULL, uses x)
#' @return `t(x) %*% y` as an mlx object
#' @param ... Additional arguments passed to base::crossprod.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.matmul}
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
#' @inheritParams mlx_matrix_required
#' @param y An mlx matrix (default: NULL, uses x)
#' @return `x %*% t(y)` as an mlx object
#' @param ... Additional arguments passed to base::tcrossprod.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.matmul}
#' @export
#' @method tcrossprod mlx
#' @examples
#' x <- as_mlx(matrix(1:6, 2, 3))
#' tcrossprod(x)
tcrossprod.mlx <- function(x, y = NULL, ...) {
  if (is.null(y)) y <- x
  x %*% t(y)
}

#' Reduce mlx array over all axes
#'
#' @param x mlx array.
#' @param op Character string naming the reduction.
#' @param ddof Integer delta degrees of freedom.
#' @return mlx scalar array.
#' @noRd
.mlx_reduce <- function(x, op, ddof = 0L) {
  ptr <- cpp_mlx_reduce(x$ptr, op, as.integer(ddof))
  .mlx_wrap_result(ptr, x$device)
}

#' Reduce mlx array along single axis
#'
#' @param x mlx array.
#' @param op Character string naming the reduction.
#' @param axis Integer (1-indexed).
#' @param drop Logical indicating whether to drop the reduced axis (default `TRUE`).
#' @param ddof Integer delta degrees of freedom.
#' @return mlx array with reduced axis.
#' @noRd
.mlx_reduce_axis <- function(x, op, axis, drop, ddof = 0L) {
  axis0 <- as.integer(axis) - 1L
  if (axis0 < 0L || axis0 >= length(x$dim)) {
    stop("axis is out of bounds for input array", call. = FALSE)
  }
  ptr <- cpp_mlx_reduce_axis(x$ptr, op, axis0, !isTRUE(drop), as.integer(ddof))
  .mlx_wrap_result(ptr, x$device)
}

#' Reduce mlx array along multiple axes
#'
#' @param x mlx array.
#' @param op Character string naming the reduction.
#' @param axes Integer vector of 1-indexed axes.
#' @param drop Logical controlling dimension dropping.
#' @param ddof Integer delta degrees of freedom.
#' @return mlx array with reduced axes.
#' @noRd
.mlx_reduce_axes <- function(x, op, axes, drop, ddof = 0L) {
  axes <- as.integer(axes)
  if (any(is.na(axes))) {
    stop("axis must be a vector of integers", call. = FALSE)
  }
  ndim <- length(x$dim)
  if (any(axes < 1L | axes > ndim)) {
    stop("axis is out of bounds for input array", call. = FALSE)
  }
  axes <- unique(axes)
  if (!drop) {
    for (ax in sort(axes)) {
      x <- .mlx_reduce_axis(x, op, axis = ax, drop = FALSE, ddof = ddof)
    }
  } else {
    for (ax in sort(axes, decreasing = TRUE)) {
      x <- .mlx_reduce_axis(x, op, axis = ax, drop = TRUE, ddof = ddof)
    }
  }
  x
}

#' Dispatch reduction to appropriate handler
#'
#' @param x mlx array or coercible object.
#' @param op Character string naming the reduction.
#' @param axis Integer vector of axes or NULL.
#' @param drop Logical controlling dimension dropping.
#' @param ddof Integer delta degrees of freedom.
#' @return mlx array with reduction result.
#' @noRd
.mlx_reduce_dispatch <- function(x, op, axis = NULL, drop = TRUE, ddof = 0L) {
  x <- if (inherits(x, "mlx")) x else as_mlx(x)
  if (is.null(axis)) {
    return(.mlx_reduce(x, op, ddof = ddof))
  }
  if (!is.logical(drop) || length(drop) != 1L) {
    stop("drop must be a single logical value", call. = FALSE)
  }
  axes <- axis
  .mlx_reduce_axes(x, op, axes, drop = drop, ddof = ddof)
}

#' @export
Summary.mlx <- function(x, ..., na.rm = FALSE) {
  op <- .Generic
  if (!(op %in% c("sum", "prod", "min", "max", "all", "any"))) {
    stop("Operation not implemented for mlx objects: ", op, call. = FALSE)
  }
  if (na.rm) {
    warning("na.rm is ignored for mlx arrays", call. = FALSE)
  }

  dots <- list(...)
  axis <- dots$axis
  drop_arg <- dots$drop
  if (!is.null(axis)) dots$axis <- NULL
  if (!is.null(drop_arg)) dots$drop <- NULL

  args <- c(list(x), dots)

  # If axis/drop specified, limit to single operand
  if (!is.null(axis) || !is.null(drop_arg)) {
    if (length(args) > 1L) {
      stop("axis/drop arguments are only supported when reducing a single array", call. = FALSE)
    }
    drop_val <- if (is.null(drop_arg)) TRUE else drop_arg
    return(.mlx_reduce_dispatch(args[[1L]], switch(op,
      sum = "sum",
      prod = "prod",
      min = "min",
      max = "max",
      all = "all",
      any = "any"
    ), axis = axis, drop = drop_val))
  }

  reduce_one <- function(obj) {
    obj_mlx <- if (inherits(obj, "mlx")) obj else as_mlx(obj)
    .mlx_reduce(obj_mlx, switch(op,
      sum = "sum",
      prod = "prod",
      min = "min",
      max = "max",
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
        min = mlx_minimum(result, scalar),
        max = mlx_maximum(result, scalar),
        all = result & scalar,
        any = result | scalar
      )
    }
  }
  result
}

#' Cumulative sum and product
#'
#' Compute cumulative sums or products along an axis.
#'
#' @inheritParams mlx_array_required
#' @param axis Optional axis along which to compute cumulative operation.
#'   If `NULL` (default), the array is flattened first.
#' @param reverse If `TRUE`, compute in reverse order.
#' @param inclusive If `TRUE` (default), include the current element in the cumulative operation.
#'   If `FALSE`, the cumulative operation is exclusive (starts from identity element).
#' @return An mlx array with cumulative sums or products.
#' @seealso [cumsum()], [cumprod()],
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.cumsum},
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.cumprod}
#' @export
#' @examples
#' x <- as_mlx(1:5)
#' mlx_cumsum(x)  # [1, 3, 6, 10, 15]
#'
#' mat <- as_mlx(matrix(1:12, 3, 4))
#' mlx_cumsum(mat, axis = 1)  # cumsum down rows
mlx_cumsum <- function(x, axis = NULL, reverse = FALSE, inclusive = TRUE) {
  x <- as_mlx(x)

  axis_mlx <- .mlx_normalize_axis(axis, x)

  ptr <- cpp_mlx_cumsum(x$ptr, axis_mlx, reverse, inclusive)
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_cumsum
#' @export
mlx_cumprod <- function(x, axis = NULL, reverse = FALSE, inclusive = TRUE) {
  x <- as_mlx(x)

  axis_mlx <- .mlx_normalize_axis(axis, x)

  ptr <- cpp_mlx_cumprod(x$ptr, axis_mlx, reverse, inclusive)
  .mlx_wrap_result(ptr, x$device)
}
