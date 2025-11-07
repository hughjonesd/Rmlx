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
#'   [mlx.core.sum](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.sum),
#'   [mlx.core.prod](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.prod),
#'   [mlx.core.all](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.all),
#'   [mlx.core.any](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.any),
#'   [mlx.core.mean](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.mean),
#'   [mlx.core.var](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.var),
#'   [mlx.core.std](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.std)
#' @details
#' `mlx_all()` and `mlx_any()` return mlx boolean scalars, while the
#' base R reducers [all()] and [any()] applied to mlx inputs return plain
#' logical scalars.
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
#' @seealso [mlx.core.mean](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.mean)
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
#' @seealso [mlx.core.mean](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.mean)
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 3, 2))
#' rowMeans(x)
rowMeans <- function(x, ...) {
  UseMethod("rowMeans")
}

#' @rdname rowMeans
#' @export
#' @method rowMeans default
rowMeans.default <- function(x, na.rm = FALSE, dims = 1, ...) {
  base::rowMeans(x, na.rm = na.rm, dims = dims, ...)
}

#' @rdname rowMeans
#' @export
#' @method rowMeans mlx
rowMeans.mlx <- function(x, na.rm = FALSE, dims = 1, ...) {
  .mlx_reduce_axis(x, "mean", axis = 2L, drop = TRUE)
}

#' Column means for mlx arrays
#'
#' @inheritParams mlx_reduction_base
#' @return An mlx array if `x` is mlx, otherwise a numeric vector.
#' @seealso [mlx.core.mean](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.mean)
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 3, 2))
#' colMeans(x)
colMeans <- function(x, ...) {
  UseMethod("colMeans")
}

#' @rdname colMeans
#' @export
#' @method colMeans default
colMeans.default <- function(x, na.rm = FALSE, dims = 1, ...) {
  base::colMeans(x, na.rm = na.rm, dims = dims, ...)
}

#' @rdname colMeans
#' @export
#' @method colMeans mlx
colMeans.mlx <- function(x, na.rm = FALSE, dims = 1, ...) {
  .mlx_reduce_axis(x, "mean", axis = 1L, drop = TRUE)
}

#' Row sums for mlx arrays
#'
#' @inheritParams mlx_reduction_base
#' @return An mlx array if `x` is mlx, otherwise a numeric vector.
#' @seealso [mlx.core.sum](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.sum)
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 3, 2))
#' rowSums(x)
rowSums <- function(x, ...) {
  UseMethod("rowSums")
}

#' @rdname rowSums
#' @export
#' @method rowSums default
rowSums.default <- function(x, na.rm = FALSE, dims = 1, ...) {
  base::rowSums(x, na.rm = na.rm, dims = dims, ...)
}

#' @rdname rowSums
#' @export
#' @method rowSums mlx
rowSums.mlx <- function(x, na.rm = FALSE, dims = 1, ...) {
  .mlx_reduce_axis(x, "sum", axis = 2L, drop = TRUE)
}

#' Column sums for mlx arrays
#'
#' @inheritParams mlx_reduction_base
#' @return An mlx array if `x` is mlx, otherwise a numeric vector.
#' @seealso [mlx.core.sum](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.sum)
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 3, 2))
#' colSums(x)
colSums <- function(x, ...) {
  UseMethod("colSums")
}

#' @rdname colSums
#' @export
#' @method colSums default
colSums.default <- function(x, na.rm = FALSE, dims = 1, ...) {
  base::colSums(x, na.rm = na.rm, dims = dims, ...)
}

#' @rdname colSums
#' @export
#' @method colSums mlx
colSums.mlx <- function(x, na.rm = FALSE, dims = 1, ...) {
  .mlx_reduce_axis(x, "sum", axis = 1L, drop = TRUE)
}

#' Transpose of MLX matrix
#'
#' @inheritParams mlx_matrix_required
#' @return Transposed mlx matrix
#' @seealso [mlx.core.transpose](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.transpose)
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
#' @seealso [mlx.core.matmul](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.matmul)
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
#' @seealso [mlx.core.matmul](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.matmul)
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
    res <- .mlx_reduce_dispatch(args[[1L]], switch(op,
      sum = "sum",
      prod = "prod",
      min = "min",
      max = "max",
      all = "all",
      any = "any"
    ), axis = axis, drop = drop_val)
    if (op %in% c("all", "any")) {
      return(as.logical(as.matrix(res)))
    }
    return(res)
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
  if (op %in% c("all", "any")) {
    return(as.logical(as.matrix(result)))
  }
  result
}

#' Row and column indices for mlx arrays
#'
#' Extends base [row()] and [col()] so they also accept mlx arrays. When
#' `as.factor = FALSE` the result stays on the MLX backend, avoiding
#' round-tripping through base R.
#'
#' @inheritParams base::row
#' @return A matrix or array of row indices (for `row()`) or column indices
#'   (for `col()`), matching the base R behaviour.
#' @export
row <- function(x, as.factor = FALSE) {
  UseMethod("row")
}

#' @rdname row
#' @export
row.default <- function(x, as.factor = FALSE) {
  base::row(x, as.factor = as.factor)
}

#' @rdname row
#' @export
row.mlx <- function(x, as.factor = FALSE) {
  dims <- dim(x)
  if (length(dims) <= 1L) {
    stop("a matrix-like object is required as argument to 'row'", call. = FALSE)
  }
  if (isTRUE(as.factor)) {
    warning("row.mlx() ignores as.factor = TRUE and returns mlx indices.", call. = FALSE)
  }
  rows <- mlx_arange(
    dims[1] + 1,
    start = 1,
    dtype = "int32",
    device = x$device
  )
  target_shape <- c(dims[1], rep.int(1L, length(dims) - 1L))
  reshaped <- mlx_reshape(rows, target_shape)
  mlx_broadcast_to(reshaped, dims)
}

#' @rdname row
#' @export
col <- function(x, as.factor = FALSE) {
  UseMethod("col")
}

#' @rdname row
#' @export
col.default <- function(x, as.factor = FALSE) {
  base::col(x, as.factor = as.factor)
}

#' @rdname row
#' @export
col.mlx <- function(x, as.factor = FALSE) {
  dims <- dim(x)
  if (length(dims) <= 1L) {
    stop("a matrix-like object is required as argument to 'col'", call. = FALSE)
  }
  if (isTRUE(as.factor)) {
    warning("col.mlx() ignores as.factor = TRUE and returns mlx indices.", call. = FALSE)
  }
  cols <- mlx_arange(
    dims[2] + 1,
    start = 1,
    dtype = "int32",
    device = x$device
  )
  target_shape <- c(1L, dims[2], rep.int(1L, length(dims) - 2L))
  reshaped <- mlx_reshape(cols, target_shape)
  mlx_broadcast_to(reshaped, dims)
}

#' Scale mlx arrays
#'
#' Extends base [scale()] to handle mlx inputs without moving data back to
#' base R. The computation delegates to MLX reductions and broadcasting. When
#' centering or scaling values are computed, the attributes `"scaled:center"`
#' and `"scaled:scale"` are stored as 1 x `ncol(x)` mlx arrays (user-supplied
#' numeric vectors are preserved as-is). These attributes remain MLX arrays even
#' after coercing with [as.matrix()], so they stay lazily evaluated.
#'
#' @inheritParams base::scale
#' @return An mlx array with centred/scaled columns.
#' @exportS3Method scale mlx
scale.mlx <- function(x, center = TRUE, scale = TRUE) {
  x_mlx <- as_mlx(x)
  if (length(x_mlx$dim) != 2L) {
    stop("scale.mlx() currently supports 2D arrays (matrices).", call. = FALSE)
  }

  n_rows <- x_mlx$dim[1L]
  n_cols <- x_mlx$dim[2L]
  result <- x_mlx
  center_attr <- NULL
  scale_attr <- NULL

  # Centering
  if (!identical(center, FALSE)) {
    if (isTRUE(center)) {
      centers <- mlx_mean(result, axis = 1L, drop = FALSE)
      center_attr <- centers
    } else {
      center_vec <- as.numeric(center)
      if (length(center_vec) == 1L) {
        center_vec <- rep(center_vec, n_cols)
      }
      if (length(center_vec) != n_cols) {
        stop("length of 'center' must equal the number of columns of 'x'", call. = FALSE)
      }
      center_attr <- center_vec
      centers <- as_mlx(matrix(center_vec, nrow = 1L),
                        dtype = result$dtype,
                        device = result$device)
    }
    result <- result - centers
  }

  # Scaling
  if (!identical(scale, FALSE)) {
    if (isTRUE(scale)) {
      ddof <- if (n_rows > 1L) 1L else 0L
      scales <- mlx_std(result, axis = 1L, drop = FALSE, ddof = ddof)
      scale_attr <- scales
    } else {
      scale_vec <- as.numeric(scale)
      if (length(scale_vec) == 1L) {
        scale_vec <- rep(scale_vec, n_cols)
      }
      if (length(scale_vec) != n_cols) {
        stop("length of 'scale' must equal the number of columns of 'x'", call. = FALSE)
      }
      scale_attr <- scale_vec
      scales <- as_mlx(matrix(scale_vec, nrow = 1L),
                       dtype = result$dtype,
                       device = result$device)
    }

    result <- result / scales
  }

  if (!is.null(center_attr)) {
    attr(result, "scaled:center") <- center_attr
  }
  if (!is.null(scale_attr)) {
    attr(result, "scaled:scale") <- scale_attr
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
#'   [mlx.core.cumsum](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.cumsum),
#'   [mlx.core.cumprod](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.cumprod)
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
