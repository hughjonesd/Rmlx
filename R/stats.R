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

#' Summary operations for MLX arrays
#'
#' S3 group generic for summary functions including `sum()`, `prod()`, `min()`, `max()`, `all()`, and `any()`.
#'
#' @param x mlx array or object coercible to mlx
#' @param ... Additional mlx arrays (for reducing multiple arrays), or named arguments `axis` and `drop`
#' @param na.rm Logical; currently ignored for mlx arrays (generates warning if TRUE)
#' @return An mlx array with the summary result
#' @seealso [mlx.core.array](https://ml-explore.github.io/mlx/build/html/python/array.html)
#' @aliases sum.mlx prod.mlx min.mlx max.mlx all.mlx any.mlx
#' @export
#' @method Summary mlx
#' @examples
#' x <- as_mlx(matrix(1:6, 2, 3))
#' sum(x)
#' any(x > 3)
#' all(x > 0)
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

#' Normal distribution functions
#'
#' Compute density (`mlx_dnorm`), cumulative distribution (`mlx_pnorm`),
#' and quantile (`mlx_qnorm`) functions for the normal distribution using MLX.
#'
#' @param x Vector of quantiles (mlx array or coercible to mlx)
#' @param mean Mean of the distribution (default: 0)
#' @param sd Standard deviation of the distribution (default: 1)
#' @param log If `TRUE`, return log density for `mlx_dnorm` (default: `FALSE`)
#' @inheritParams common_params
#' @return An mlx array with the computed values
#' @seealso [mlx_erf()], [mlx_erfinv()],
#'   [mlx.core.erf](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erf.html),
#'   [mlx.core.erfinv](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erfinv.html)
#' @export
#' @examples
#' x <- as_mlx(seq(-3, 3, by = 0.5))
#' as.matrix(mlx_dnorm(x))
#' as.matrix(mlx_pnorm(x))
#'
#' p <- as_mlx(c(0.025, 0.5, 0.975))
#' as.matrix(mlx_qnorm(p))
mlx_dnorm <- function(x, mean = 0, sd = 1, log = FALSE, device = mlx_default_device()) {
  x <- as_mlx(x, device = device)

  if (sd <= 0) {
    stop("sd must be positive", call. = FALSE)
  }

  # Convert mean and sd to mlx arrays
  mean_mlx <- as_mlx(mean, device = device)
  sd_mlx <- as_mlx(sd, device = device)

  # Standardize
  z <- (x - mean_mlx) / sd_mlx

  # dnorm(x, mean, sd) = (1/sqrt(2*pi*sd^2)) * exp(-0.5 * z^2)
  # = (1/(sd * sqrt(2*pi))) * exp(-0.5 * z^2)

  sqrt_2pi <- as_mlx(sqrt(2 * pi), device = device)
  log_density <- -0.5 * z^2 - log(sd_mlx) - log(sqrt_2pi)

  if (log) {
    return(log_density)
  } else {
    return(exp(log_density))
  }
}

#' @rdname mlx_dnorm
#' @export
#' @param p Vector of probabilities (mlx array or coercible to mlx)
mlx_pnorm <- function(x, mean = 0, sd = 1, device = mlx_default_device()) {
  x <- as_mlx(x, device = device)

  if (sd <= 0) {
    stop("sd must be positive", call. = FALSE)
  }

  # Convert mean and sd to mlx arrays
  mean_mlx <- as_mlx(mean, device = device)
  sd_mlx <- as_mlx(sd, device = device)

  # Standardize
  z <- (x - mean_mlx) / sd_mlx

  # pnorm(x) = 0.5 * (1 + erf(z / sqrt(2)))
  sqrt_2 <- as_mlx(sqrt(2), device = device)
  return(0.5 * (1 + mlx_erf(z / sqrt_2)))
}

#' @rdname mlx_dnorm
#' @export
mlx_qnorm <- function(p, mean = 0, sd = 1, device = mlx_default_device()) {
  p <- as_mlx(p, device = device)

  if (sd <= 0) {
    stop("sd must be positive", call. = FALSE)
  }

  # Convert mean and sd to mlx arrays
  mean_mlx <- as_mlx(mean, device = device)
  sd_mlx <- as_mlx(sd, device = device)

  # qnorm(p) = mean + sd * sqrt(2) * erfinv(2*p - 1)
  sqrt_2 <- as_mlx(sqrt(2), device = device)
  return(mean_mlx + sd_mlx * sqrt_2 * mlx_erfinv(2 * p - 1))
}

#' Uniform distribution functions
#'
#' Compute density (`mlx_dunif`), cumulative distribution (`mlx_punif`),
#' and quantile (`mlx_qunif`) functions for the uniform distribution using MLX.
#'
#' @param x Vector of quantiles (mlx array or coercible to mlx)
#' @param min,max Lower and upper limits of the distribution (default: 0, 1)
#' @param log If `TRUE`, return log density for `mlx_dunif` (default: `FALSE`)
#' @inheritParams common_params
#' @return An mlx array with the computed values
#' @export
#' @examples
#' x <- as_mlx(seq(0, 1, by = 0.1))
#' as.matrix(mlx_dunif(x))
#' as.matrix(mlx_punif(x))
#'
#' p <- as_mlx(c(0.25, 0.5, 0.75))
#' as.matrix(mlx_qunif(p))
mlx_dunif <- function(x, min = 0, max = 1, log = FALSE, device = mlx_default_device()) {
  x <- as_mlx(x, device = device)

  if (min >= max) {
    stop("min must be less than max", call. = FALSE)
  }

  min_mlx <- as_mlx(min, device = device)
  max_mlx <- as_mlx(max, device = device)

  # dunif(x, min, max) = 1/(max-min) for x in [min, max], 0 otherwise
  width <- max_mlx - min_mlx
  in_range <- (x >= min_mlx) & (x <= max_mlx)

  if (log) {
    # log(1/width) = -log(width)
    log_density <- -log(width)
    # Set out of range to -Inf
    result <- mlx_where(in_range, log_density, as_mlx(-Inf, device = device))
  } else {
    density <- 1 / width
    result <- mlx_where(in_range, density, as_mlx(0, device = device))
  }

  return(result)
}

#' @rdname mlx_dunif
#' @export
#' @param p Vector of probabilities (mlx array or coercible to mlx)
mlx_punif <- function(x, min = 0, max = 1, device = mlx_default_device()) {
  x <- as_mlx(x, device = device)

  if (min >= max) {
    stop("min must be less than max", call. = FALSE)
  }

  min_mlx <- as_mlx(min, device = device)
  max_mlx <- as_mlx(max, device = device)

  # punif(x, min, max) = (x-min)/(max-min), clipped to [0, 1]
  prob <- (x - min_mlx) / (max_mlx - min_mlx)
  return(mlx_clip(prob, 0, 1))
}

#' @rdname mlx_dunif
#' @export
mlx_qunif <- function(p, min = 0, max = 1, device = mlx_default_device()) {
  p <- as_mlx(p, device = device)

  if (min >= max) {
    stop("min must be less than max", call. = FALSE)
  }

  min_mlx <- as_mlx(min, device = device)
  max_mlx <- as_mlx(max, device = device)

  # qunif(p, min, max) = min + p*(max-min)
  return(min_mlx + p * (max_mlx - min_mlx))
}

#' Exponential distribution functions
#'
#' Compute density (`mlx_dexp`), cumulative distribution (`mlx_pexp`),
#' and quantile (`mlx_qexp`) functions for the exponential distribution using MLX.
#'
#' @param x Vector of quantiles (mlx array or coercible to mlx)
#' @param rate Rate parameter (default: 1)
#' @param log If `TRUE`, return log density for `mlx_dexp` (default: `FALSE`)
#' @inheritParams common_params
#' @return An mlx array with the computed values
#' @export
#' @examples
#' x <- as_mlx(seq(0, 5, by = 0.5))
#' as.matrix(mlx_dexp(x))
#' as.matrix(mlx_pexp(x))
#'
#' p <- as_mlx(c(0.25, 0.5, 0.75))
#' as.matrix(mlx_qexp(p))
mlx_dexp <- function(x, rate = 1, log = FALSE, device = mlx_default_device()) {
  x <- as_mlx(x, device = device)

  if (rate <= 0) {
    stop("rate must be positive", call. = FALSE)
  }

  rate_mlx <- as_mlx(rate, device = device)

  # dexp(x, rate) = rate * exp(-rate*x) for x >= 0, 0 otherwise
  non_negative <- x >= 0

  if (log) {
    # log(rate * exp(-rate*x)) = log(rate) - rate*x
    log_density <- log(rate_mlx) - rate_mlx * x
    result <- mlx_where(non_negative, log_density, as_mlx(-Inf, device = device))
  } else {
    density <- rate_mlx * exp(-rate_mlx * x)
    result <- mlx_where(non_negative, density, as_mlx(0, device = device))
  }

  return(result)
}

#' @rdname mlx_dexp
#' @export
#' @param p Vector of probabilities (mlx array or coercible to mlx)
mlx_pexp <- function(x, rate = 1, device = mlx_default_device()) {
  x <- as_mlx(x, device = device)

  if (rate <= 0) {
    stop("rate must be positive", call. = FALSE)
  }

  rate_mlx <- as_mlx(rate, device = device)

  # pexp(x, rate) = 1 - exp(-rate*x) for x >= 0, 0 otherwise
  prob <- 1 - exp(-rate_mlx * x)
  non_negative <- x >= 0
  return(mlx_where(non_negative, prob, as_mlx(0, device = device)))
}

#' @rdname mlx_dexp
#' @export
mlx_qexp <- function(p, rate = 1, device = mlx_default_device()) {
  p <- as_mlx(p, device = device)

  if (rate <= 0) {
    stop("rate must be positive", call. = FALSE)
  }

  rate_mlx <- as_mlx(rate, device = device)

  # qexp(p, rate) = -log(1-p) / rate
  return(-log(1 - p) / rate_mlx)
}

#' Lognormal distribution functions
#'
#' Compute density (`mlx_dlnorm`), cumulative distribution (`mlx_plnorm`),
#' and quantile (`mlx_qlnorm`) functions for the lognormal distribution using MLX.
#'
#' @param x Vector of quantiles (mlx array or coercible to mlx)
#' @param meanlog,sdlog Mean and standard deviation of distribution on log scale
#'   (default: 0, 1)
#' @param log If `TRUE`, return log density for `mlx_dlnorm` (default: `FALSE`)
#' @inheritParams common_params
#' @return An mlx array with the computed values
#' @export
#' @examples
#' x <- as_mlx(seq(0.1, 3, by = 0.2))
#' as.matrix(mlx_dlnorm(x))
#' as.matrix(mlx_plnorm(x))
#'
#' p <- as_mlx(c(0.25, 0.5, 0.75))
#' as.matrix(mlx_qlnorm(p))
mlx_dlnorm <- function(x, meanlog = 0, sdlog = 1, log = FALSE,
                       device = mlx_default_device()) {
  x <- as_mlx(x, device = device)

  if (sdlog <= 0) {
    stop("sdlog must be positive", call. = FALSE)
  }

  # dlnorm(x) = dnorm(log(x), meanlog, sdlog) / x for x > 0
  positive <- x > 0

  log_x <- log(x)
  log_dnorm <- mlx_dnorm(log_x, mean = meanlog, sd = sdlog, log = TRUE, device = device)

  if (log) {
    # log(dnorm(log(x)) / x) = log(dnorm(log(x))) - log(x)
    log_density <- log_dnorm - log_x
    result <- mlx_where(positive, log_density, as_mlx(-Inf, device = device))
  } else {
    density <- exp(log_dnorm) / x
    result <- mlx_where(positive, density, as_mlx(0, device = device))
  }

  return(result)
}

#' @rdname mlx_dlnorm
#' @export
#' @param p Vector of probabilities (mlx array or coercible to mlx)
mlx_plnorm <- function(x, meanlog = 0, sdlog = 1, device = mlx_default_device()) {
  x <- as_mlx(x, device = device)

  if (sdlog <= 0) {
    stop("sdlog must be positive", call. = FALSE)
  }

  # plnorm(x) = pnorm(log(x), meanlog, sdlog) for x > 0, 0 otherwise
  positive <- x > 0
  prob <- mlx_pnorm(log(x), mean = meanlog, sd = sdlog, device = device)
  return(mlx_where(positive, prob, as_mlx(0, device = device)))
}

#' @rdname mlx_dlnorm
#' @export
mlx_qlnorm <- function(p, meanlog = 0, sdlog = 1, device = mlx_default_device()) {
  p <- as_mlx(p, device = device)

  if (sdlog <= 0) {
    stop("sdlog must be positive", call. = FALSE)
  }

  # qlnorm(p) = exp(qnorm(p, meanlog, sdlog))
  return(exp(mlx_qnorm(p, mean = meanlog, sd = sdlog, device = device)))
}

#' Logistic distribution functions
#'
#' Compute density (`mlx_dlogis`), cumulative distribution (`mlx_plogis`),
#' and quantile (`mlx_qlogis`) functions for the logistic distribution using MLX.
#'
#' @param x Vector of quantiles (mlx array or coercible to mlx)
#' @param location,scale Location and scale parameters (default: 0, 1)
#' @param log If `TRUE`, return log density for `mlx_dlogis` (default: `FALSE`)
#' @inheritParams common_params
#' @return An mlx array with the computed values
#' @export
#' @examples
#' x <- as_mlx(seq(-3, 3, by = 0.5))
#' as.matrix(mlx_dlogis(x))
#' as.matrix(mlx_plogis(x))
#'
#' p <- as_mlx(c(0.25, 0.5, 0.75))
#' as.matrix(mlx_qlogis(p))
mlx_dlogis <- function(x, location = 0, scale = 1, log = FALSE,
                       device = mlx_default_device()) {
  x <- as_mlx(x, device = device)

  if (scale <= 0) {
    stop("scale must be positive", call. = FALSE)
  }

  location_mlx <- as_mlx(location, device = device)
  scale_mlx <- as_mlx(scale, device = device)

  # Standardize
  z <- (x - location_mlx) / scale_mlx

  # dlogis(x) = exp(z) / (scale * (1 + exp(z))^2)
  # For numerical stability, use different forms for positive/negative z
  # log(dlogis) = z - log(scale) - 2*log(1 + exp(z))
  #             = z - log(scale) - 2*log1p(exp(z))  for z < 0
  #             = -log(scale) - z - 2*log1p(exp(-z)) for z >= 0

  if (log) {
    # Use log1p for better numerical stability
    exp_z <- exp(z)
    log_density <- z - log(scale_mlx) - 2 * log(1 + exp_z)
    return(log_density)
  } else {
    exp_z <- exp(z)
    density <- exp_z / (scale_mlx * (1 + exp_z)^2)
    return(density)
  }
}

#' @rdname mlx_dlogis
#' @export
#' @param p Vector of probabilities (mlx array or coercible to mlx)
mlx_plogis <- function(x, location = 0, scale = 1, device = mlx_default_device()) {
  x <- as_mlx(x, device = device)

  if (scale <= 0) {
    stop("scale must be positive", call. = FALSE)
  }

  location_mlx <- as_mlx(location, device = device)
  scale_mlx <- as_mlx(scale, device = device)

  # Standardize
  z <- (x - location_mlx) / scale_mlx

  # plogis(x) = 1 / (1 + exp(-z))
  return(1 / (1 + exp(-z)))
}

#' @rdname mlx_dlogis
#' @export
mlx_qlogis <- function(p, location = 0, scale = 1, device = mlx_default_device()) {
  p <- as_mlx(p, device = device)

  if (scale <= 0) {
    stop("scale must be positive", call. = FALSE)
  }

  location_mlx <- as_mlx(location, device = device)
  scale_mlx <- as_mlx(scale, device = device)

  # qlogis(p) = location + scale * log(p / (1-p))
  return(location_mlx + scale_mlx * log(p / (1 - p)))
}

#' Log-gamma function
#'
#' Compute the natural logarithm of the gamma function using the Lanczos
#' approximation.
#'
#' @inheritParams mlx_array_required
#' @return An mlx array with log(Γ(x))
#' @details
#' Uses the Lanczos approximation with g=7 and 9 coefficients for accuracy
#' to about 15 decimal places.
#' @export
#' @examples
#' x <- as_mlx(c(1, 2, 3, 4, 5))
#' as.matrix(mlx_lgamma(x))
mlx_lgamma <- function(x, device = mlx_default_device()) {
  x <- as_mlx(x, device = device)

  # Lanczos approximation coefficients (g = 7, n = 9)
  # These give accuracy to about 15 decimal places
  lanczos_g <- 7
  lanczos_coef <- c(
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  )

  # Convert to mlx arrays
  g <- as_mlx(lanczos_g, device = device)
  coef <- lapply(lanczos_coef, function(c) as_mlx(c, device = device))

  # For x < 0.5, use reflection formula: Γ(1-z)Γ(z) = π/sin(πz)
  # lgamma(1-z) + lgamma(z) = log(π) - log(sin(πz))
  use_reflection <- x < 0.5

  # Work with z = x for x >= 0.5, or z = 1-x for x < 0.5
  z <- mlx_where(use_reflection, 1 - x, x)

  # Compute Lanczos approximation for z >= 0.5
  # lgamma(z) = 0.5*log(2π) + (z-0.5)*log(z+g-0.5) - (z+g-0.5) + log(Ag(z))

  # Compute A_g(z) = c0 + c1/(z) + c2/(z+1) + ... + c8/(z+7)
  ag <- coef[[1]]
  for (i in 2:length(coef)) {
    ag <- ag + coef[[i]] / (z + as_mlx(i - 2, device = device))
  }

  # Compute lgamma(z)
  log_sqrt_2pi <- as_mlx(0.5 * log(2 * pi), device = device)
  zgh <- z + g - 0.5
  result <- log_sqrt_2pi + (z - 0.5) * log(zgh) - zgh + log(ag)

  # Apply reflection formula if needed
  # lgamma(x) = log(π) - log(sin(πx)) - lgamma(1-x)
  pi_mlx <- as_mlx(pi, device = device)
  reflected <- log(pi_mlx) - log(abs(sin(pi_mlx * x))) - result

  return(mlx_where(use_reflection, reflected, result))
}

#' Regularized incomplete beta function
#'
#' Compute the regularized incomplete beta function I_x(a,b) using continued
#' fractions.
#'
#' @param x Upper limit of integration (between 0 and 1)
#' @param a,b Shape parameters (must be positive)
#' @inheritParams common_params
#' @return An mlx array with I_x(a,b)
#' @details
#' The regularized incomplete beta function is defined as:
#' \deqn{I_x(a,b) = \frac{B_x(a,b)}{B(a,b)} = \frac{1}{B(a,b)} \int_0^x t^{a-1}(1-t)^{b-1}dt}
#'
#' Uses Lentz's algorithm for continued fraction evaluation, which typically
#' converges in less than 100 iterations.
#' @keywords internal
mlx_betainc <- function(x, a, b, device = mlx_default_device()) {
  x <- as_mlx(x, device = device)
  a <- as_mlx(a, device = device)
  b <- as_mlx(b, device = device)

  # Use symmetry for better convergence: I_x(a,b) = 1 - I_{1-x}(b,a)
  # Switch if x > (a+1)/(a+b+2)
  swap_threshold <- (a + 1) / (a + b + 2)
  use_symmetry <- x > swap_threshold

  x_work <- mlx_where(use_symmetry, 1 - x, x)
  a_work <- mlx_where(use_symmetry, b, a)
  b_work <- mlx_where(use_symmetry, a, b)

  # Use the standard continued fraction representation
  # I_x(a,b) = [x^a * (1-x)^b * CF] / [a * B(a,b)]
  # where CF = 1/(1 + d₁/(1 + d₂/(1 + d₃/...)))

  log_beta <- mlx_lgamma(a_work, device) + mlx_lgamma(b_work, device) -
    mlx_lgamma(a_work + b_work, device)

  front <- a_work * log(x_work) + b_work * log(1 - x_work) - log(a_work) - log_beta

  # Modified Lentz algorithm for continued fraction
  tiny <- as_mlx(1e-30, device = device)
  one <- as_mlx(1, device = device)

  f <- one
  c <- one
  d <- zero <- as_mlx(0, device = device)

  for (m in 0:150) {
    m_mlx <- as_mlx(m, device = device)

    # Compute coefficient d_m
    if (m == 0) {
      # First term d_0 = 1
      dm <- one
    } else {
      # For m >= 1, alternate between two formulas
      if (m %% 2 == 1) {
        # Odd m: d_m = - (a+m')(a+b+m')x / ((a+2m')(a+2m'+1))
        # where m' = (m-1)/2
        mp <- (m - 1) / 2
        mp_mlx <- as_mlx(mp, device = device)
        dm <- -(a_work + mp_mlx) * (a_work + b_work + mp_mlx) * x_work /
              ((a_work + 2*mp_mlx) * (a_work + 2*mp_mlx + 1))
      } else {
        # Even m: d_m = m'(b-m')x / ((a+2m'-1)(a+2m'))
        # where m' = m/2
        mp <- m / 2
        mp_mlx <- as_mlx(mp, device = device)
        dm <- mp_mlx * (b_work - mp_mlx) * x_work /
              ((a_work + 2*mp_mlx - 1) * (a_work + 2*mp_mlx))
      }
    }

    # Lentz update
    d <- one / (one + dm * d)
    d <- mlx_where(abs(d) < tiny, tiny, d)

    c <- one + dm / c
    c <- mlx_where(abs(c) < tiny, tiny, c)

    f <- f * c * d
  }

  result <- exp(front) * f

  # Apply symmetry if we swapped
  result <- mlx_where(use_symmetry, 1 - result, result)

  # Clip to [0, 1] for numerical safety
  return(mlx_clip(result, 0, 1))
}

#' Student's t distribution functions
#'
#' Compute density (`mlx_dt`), cumulative distribution (`mlx_pt`),
#' and quantile (`mlx_qt`) functions for Student's t distribution using MLX.
#'
#' @param x Vector of quantiles (mlx array or coercible to mlx)
#' @param df Degrees of freedom (must be positive)
#' @param log If `TRUE`, return log density for `mlx_dt` (default: `FALSE`)
#' @inheritParams common_params
#' @return An mlx array with the computed values
#' @details
#' The t distribution with df degrees of freedom has density:
#' \deqn{f(x) = \frac{\Gamma((df+1)/2)}{\sqrt{df\pi}\Gamma(df/2)}
#'   (1 + x^2/df)^{-(df+1)/2}}
#'
#' For the CDF and quantile functions, approximations are used that become
#' more accurate as df increases. For df > 30, the normal approximation is used.
#' @export
#' @examples
#' x <- as_mlx(seq(-3, 3, by = 0.5))
#' as.matrix(mlx_dt(x, df = 5))
#' as.matrix(mlx_pt(x, df = 5))
#'
#' p <- as_mlx(c(0.025, 0.5, 0.975))
#' as.matrix(mlx_qt(p, df = 5))
mlx_dt <- function(x, df, log = FALSE, device = mlx_default_device()) {
  x <- as_mlx(x, device = device)

  if (df <= 0) {
    stop("df must be positive", call. = FALSE)
  }

  df_mlx <- as_mlx(df, device = device)

  # dt(x, df) = Γ((df+1)/2) / (sqrt(df*π) * Γ(df/2)) * (1 + x²/df)^(-(df+1)/2)
  # log(dt) = lgamma((df+1)/2) - lgamma(df/2) - 0.5*log(df*π) - ((df+1)/2)*log(1 + x²/df)

  lgamma_half_dfp1 <- mlx_lgamma((df_mlx + 1) / 2, device = device)
  lgamma_half_df <- mlx_lgamma(df_mlx / 2, device = device)
  log_sqrt_df_pi <- 0.5 * log(df_mlx * as_mlx(pi, device = device))

  u <- 1 + x^2 / df_mlx
  log_density <- lgamma_half_dfp1 - lgamma_half_df - log_sqrt_df_pi -
    ((df_mlx + 1) / 2) * log(u)

  if (log) {
    return(log_density)
  } else {
    return(exp(log_density))
  }
}

#' @rdname mlx_dt
#' @export
#' @param p Vector of probabilities (mlx array or coercible to mlx)
mlx_pt <- function(x, df, device = mlx_default_device()) {
  x <- as_mlx(x, device = device)

  if (df <= 0) {
    stop("df must be positive", call. = FALSE)
  }

  df_mlx <- as_mlx(df, device = device)

  # Use relationship with incomplete beta function:
  # For t > 0: pt(t, df) = 1 - 0.5 * I_x(df/2, 0.5)
  # For t < 0: use symmetry pt(-t, df) = 1 - pt(t, df)
  # where x = df / (df + t²)

  x_abs <- abs(x)
  is_negative <- x < 0

  # Compute x for incomplete beta: x = df / (df + t²)
  beta_x <- df_mlx / (df_mlx + x_abs^2)

  # Compute I_x(df/2, 0.5)
  a <- df_mlx / 2
  b <- as_mlx(0.5, device = device)
  ibeta <- mlx_betainc(beta_x, a, b, device = device)

  # pt(|t|, df) = 1 - 0.5 * I_x(df/2, 0.5) for t > 0
  prob_positive <- 1 - 0.5 * ibeta

  # Use symmetry for negative values
  prob <- mlx_where(is_negative, 1 - prob_positive, prob_positive)

  return(prob)
}

#' @rdname mlx_dt
#' @export
mlx_qt <- function(p, df, device = mlx_default_device()) {
  p <- as_mlx(p, device = device)

  if (df <= 0) {
    stop("df must be positive", call. = FALSE)
  }

  # For large df, use normal approximation
  if (df > 30) {
    return(mlx_qnorm(p, device = device))
  }

  df_mlx <- as_mlx(df, device = device)

  # Use inverse of the Wilson-Hilferty approximation
  z <- mlx_qnorm(p, device = device)

  if (df > 2) {
    variance_factor <- sqrt(df_mlx / (df_mlx - 2))
    return(z * variance_factor)
  } else {
    # For very small df, use simpler approach
    return(z)
  }
}
