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
#' mlx_sum(x, axes = 1)
#' mlx_prod(x, axes = 2, drop = FALSE)
#' mlx_all(x > 0)
#' mlx_any(x > 3)
#' mlx_mean(x, axes = 1)
#' mlx_var(x, axes = 2)
#' mlx_std(x, ddof = 1)
#' @aliases mlx_sum mlx_prod mlx_all mlx_any mlx_mean mlx_var mlx_std
#' @name mlx_sum
NULL

#' @rdname mlx_sum
#' @export
mlx_sum <- function(x, axes = NULL, drop = TRUE) {
  .mlx_reduce_dispatch(x, "sum", axes = axes, drop = drop)
}

#' @rdname mlx_sum
#' @export
mlx_prod <- function(x, axes = NULL, drop = TRUE) {
  .mlx_reduce_dispatch(x, "prod", axes = axes, drop = drop)
}

#' @rdname mlx_sum
#' @export
mlx_all <- function(x, axes = NULL, drop = TRUE) {
  .mlx_reduce_dispatch(x, "all", axes = axes, drop = drop)
}

#' @rdname mlx_sum
#' @export
mlx_any <- function(x, axes = NULL, drop = TRUE) {
  .mlx_reduce_dispatch(x, "any", axes = axes, drop = drop)
}

#' @rdname mlx_sum
#' @export
mlx_mean <- function(x, axes = NULL, drop = TRUE) {
  .mlx_reduce_dispatch(x, "mean", axes = axes, drop = drop)
}

#' @rdname mlx_sum
#' @export
mlx_var <- function(x, axes = NULL, drop = TRUE, ddof = 0L) {
  .mlx_reduce_dispatch(x, "var", axes = axes, drop = drop, ddof = ddof)
}

#' @rdname mlx_sum
#' @export
mlx_std <- function(x, axes = NULL, drop = TRUE, ddof = 0L) {
  .mlx_reduce_dispatch(x, "std", axes = axes, drop = drop, ddof = ddof)
}

#' Mean of MLX array elements
#'
#' @inheritParams mlx_array_required
#' @param ... Additional arguments (ignored)
#' @return An mlx scalar.
#' @seealso [mlx.core.mean](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.mean)
#' @export
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
rowMeans.default <- function(x, na.rm = FALSE, dims = 1, ...) {
  base::rowMeans(x, na.rm = na.rm, dims = dims, ...)
}

#' @rdname rowMeans
#' @export
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
colMeans.default <- function(x, na.rm = FALSE, dims = 1, ...) {
  base::colMeans(x, na.rm = na.rm, dims = dims, ...)
}

#' @rdname colMeans
#' @export
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
rowSums.default <- function(x, na.rm = FALSE, dims = 1, ...) {
  base::rowSums(x, na.rm = na.rm, dims = dims, ...)
}

#' @rdname rowSums
#' @export
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
colSums.default <- function(x, na.rm = FALSE, dims = 1, ...) {
  base::colSums(x, na.rm = na.rm, dims = dims, ...)
}

#' @rdname colSums
#' @export
colSums.mlx <- function(x, na.rm = FALSE, dims = 1, ...) {
  .mlx_reduce_axis(x, "sum", axis = 1L, drop = TRUE)
}

#' Transpose of MLX matrix
#'
#' @inheritParams mlx_matrix_required
#' @return The transposed MLX matrix.
#' @seealso [mlx.core.transpose](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.transpose)
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 2, 3))
#' t(x)
t.mlx <- function(x) {
  # Layout conversion (physical reordering) happens at boundaries during copy
  ptr <- cpp_mlx_transpose(x$ptr)
  new_mlx(ptr, x$device)
}

#' Cross product
#'
#' @inheritParams mlx_matrix_required
#' @param y An mlx matrix (default: NULL, uses x)
#' @return `t(x) %*% y` as an mlx object.
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
#' @return `x %*% t(y)` as an mlx object.
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

#' Reduce an mlx array over all axes
#'
#' @param x mlx array (or object coercible to one).
#' @param op Character string naming the reduction ("sum", "mean", etc.).
#' @param ddof Delta degrees of freedom passed to variance-like reducers.
#' @return An mlx array containing the fully reduced result.
#' @noRd
.mlx_reduce <- function(x, op, ddof = 0L) {
  ptr <- cpp_mlx_reduce(x$ptr, op, as.integer(ddof))
  new_mlx(ptr, x$device)
}

#' Reduce an mlx array along a single axis
#'
#' @param x mlx array.
#' @param op Character string naming the reduction.
#' @param axis Single 1-indexed axis to reduce.
#' @param drop Logical flag: keep (`FALSE`) or drop (`TRUE`) the reduced axis.
#' @param ddof Delta degrees of freedom for variance-like reducers.
#' @return An mlx array with the selected axis reduced.
#' @noRd
.mlx_reduce_axis <- function(x, op, axis, drop, ddof = 0L) {
  axis0 <- as.integer(axis) - 1L
  if (axis0 < 0L || axis0 >= length(dim(x))) {
    stop("axis is out of bounds for input array", call. = FALSE)
  }
  ptr <- cpp_mlx_reduce_axis(x$ptr, op, axis0, !isTRUE(drop), as.integer(ddof))
  new_mlx(ptr, x$device)
}

#' Reduce an mlx array along multiple axes
#'
#' @param x mlx array.
#' @param op Character string naming the reduction.
#' @param axes Integer vector of 1-indexed axes to reduce.
#' @param drop Logical flag: keep (`FALSE`) or drop (`TRUE`) reduced axes.
#' @param ddof Delta degrees of freedom for variance-like reducers.
#' @return An mlx array with the specified axes reduced.
#' @noRd
.mlx_reduce_axes <- function(x, op, axes, drop, ddof = 0L) {
  axes <- as.integer(axes)
  if (any(is.na(axes))) {
    stop("axis must be a vector of integers", call. = FALSE)
  }
  ndim <- length(dim(x))
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
#' @param axes Integer vector of axes or NULL.
#' @param drop Logical controlling dimension dropping.
#' @param ddof Integer delta degrees of freedom.
#' @return mlx array with reduction result.
#' @noRd
.mlx_reduce_dispatch <- function(x, op, axes = NULL, drop = TRUE, ddof = 0L) {
  x <- as_mlx(x)
  if (is.null(axes)) {
    return(.mlx_reduce(x, op, ddof = ddof))
  }
  if (!is.logical(drop) || length(drop) != 1L) {
    stop("drop must be a single logical value", call. = FALSE)
  }
  .mlx_reduce_axes(x, op, axes, drop = drop, ddof = ddof)
}

#' Summary operations for MLX arrays
#'
#' S3 group generic for summary functions including `sum()`, `prod()`, `min()`, `max()`, `all()`, and `any()`.
#'
#' @param x mlx array or object coercible to mlx
#' @param ... Additional mlx arrays (for reducing multiple arrays), or named arguments `axes` (legacy `axis`) and `drop`
#' @param na.rm Logical; currently ignored for mlx arrays (generates warning if TRUE)
#' @return An mlx array with the summary result.
#' @seealso [mlx.core.array](https://ml-explore.github.io/mlx/build/html/python/array.html)
#' @aliases sum.mlx prod.mlx min.mlx max.mlx all.mlx any.mlx
#' @export
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
  axes <- dots$axes
  drop_arg <- dots$drop
  if (!is.null(axis)) dots$axis <- NULL
  if (!is.null(axes)) dots$axes <- NULL
  if (!is.null(drop_arg)) dots$drop <- NULL

  if (!is.null(axis)) {
    if (!is.null(axes)) {
      stop("Specify only one of `axis` or `axes`.", call. = FALSE)
    }
    warning("`axis` is deprecated; use `axes` instead.", call. = FALSE)
    axes <- axis
  }

  args <- c(list(x), dots)

  # If axis/drop specified, limit to single operand
  if (!is.null(axes) || !is.null(drop_arg)) {
    if (length(args) > 1L) {
      stop("axes/drop arguments are only supported when reducing a single array", call. = FALSE)
    }
    drop_val <- if (is.null(drop_arg)) TRUE else drop_arg
    res <- .mlx_reduce_dispatch(args[[1L]], switch(op,
      sum = "sum",
      prod = "prod",
      min = "min",
      max = "max",
      all = "all",
      any = "any"
    ), axes = axes, drop = drop_val)
    if (op %in% c("all", "any")) {
      return(as.logical(res))
    }
    return(res)
  }

  reduce_one <- function(obj) {
    obj_mlx <- as_mlx(obj)
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
    return(as.logical(result))
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
  if (length(dim(x_mlx)) != 2L) {
    stop("scale.mlx() currently supports 2D arrays (matrices).", call. = FALSE)
  }

  n_rows <- dim(x_mlx)[1L]
  n_cols <- dim(x_mlx)[2L]
  result <- x_mlx
  center_attr <- NULL
  scale_attr <- NULL

  # Centering
  if (!identical(center, FALSE)) {
    if (isTRUE(center)) {
      centers <- mlx_mean(result, axes = 1L, drop = FALSE)
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
                        dtype = mlx_dtype(result),
                        device = result$device)
    }
    result <- result - centers
  }

  # Scaling
  if (!identical(scale, FALSE)) {
    if (isTRUE(scale)) {
      ddof <- if (n_rows > 1L) 1L else 0L
      scales <- mlx_std(result, axes = 1L, drop = FALSE, ddof = ddof)
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
                       dtype = mlx_dtype(result),
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
#' @inheritParams common_params
#'
#' @details When `axis` is `NULL` (default), the array is flattened before
#' computing the cumulative result.
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
  new_mlx(ptr, x$device)
}

#' @rdname mlx_cumsum
#' @export
mlx_cumprod <- function(x, axis = NULL, reverse = FALSE, inclusive = TRUE) {
  x <- as_mlx(x)

  axis_mlx <- .mlx_normalize_axis(axis, x)

  ptr <- cpp_mlx_cumprod(x$ptr, axis_mlx, reverse, inclusive)
  new_mlx(ptr, x$device)
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
#' @return An mlx array with the computed values.
#' @seealso [mlx_erf()], [mlx_erfinv()],
#'   [mlx.core.erf](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erf.html),
#'   [mlx.core.erfinv](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erfinv.html)
#' @export
#' @examples
#' x <- as_mlx(seq(-3, 3, by = 0.5))
#' mlx_dnorm(x)
#' mlx_pnorm(x)
#'
#' p <- as_mlx(c(0.025, 0.5, 0.975))
#' mlx_qnorm(p)
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
#' @return An mlx array with the computed values.
#' @export
#' @examples
#' x <- as_mlx(seq(0, 1, by = 0.1))
#' mlx_dunif(x)
#' mlx_punif(x)
#'
#' p <- as_mlx(c(0.25, 0.5, 0.75))
#' mlx_qunif(p)
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
#' @return An mlx array with the computed values.
#' @export
#' @examples
#' x <- as_mlx(seq(0, 5, by = 0.5))
#' mlx_dexp(x)
#' mlx_pexp(x)
#'
#' p <- as_mlx(c(0.25, 0.5, 0.75))
#' mlx_qexp(p)
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
#' @return An mlx array with the computed values.
#' @export
#' @examples
#' x <- as_mlx(seq(0.1, 3, by = 0.2))
#' mlx_dlnorm(x)
#' mlx_plnorm(x)
#'
#' p <- as_mlx(c(0.25, 0.5, 0.75))
#' mlx_qlnorm(p)
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
#' @return An mlx array with the computed values.
#' @export
#' @examples
#' x <- as_mlx(seq(-3, 3, by = 0.5))
#' mlx_dlogis(x)
#' mlx_plogis(x)
#'
#' p <- as_mlx(c(0.25, 0.5, 0.75))
#' mlx_qlogis(p)
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

#' Compute quantiles of MLX arrays
#'
#' Calculates sample quantiles corresponding to given probabilities using linear
#' interpolation (R's type 7 quantiles, the default in [stats::quantile()]).
#' The S3 method `quantile.mlx()` provides an interface compatible with the
#' generic [stats::quantile()].
#'
#' @inheritParams common_params
#' @param probs Numeric vector of probabilities in \[0, 1\].
#' @param axis Optional integer axis (or vector of axes) along which to compute
#'   quantiles. When `NULL` (default), quantiles are computed over the entire
#'   flattened array.
#' @param drop Logical; when `TRUE` and computing quantiles along an axis with a
#'   single probability, removes the quantile dimension of length 1. Defaults to
#'   `FALSE` to match the behavior of other reduction functions.
#' @param ... Additional arguments (currently ignored by `quantile.mlx()`).
#' @return An mlx array containing the requested quantiles. The shape depends on
#'   `probs`, `axis`, and `drop`: when `axis = NULL`, returns a scalar for a
#'   single probability or a vector for multiple probabilities. When `axis` is
#'   specified, the quantile dimension replaces the reduced axis (e.g., a `(3, 4)`
#'   matrix with `axis = 1` and 2 quantiles gives `(2, 4)`), unless `drop = TRUE`
#'   with a single probability removes that dimension.
#' @seealso
#'   [stats::quantile()],
#'   [mlx.core.sort](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.sort)
#' @details
#' Uses type 7 quantiles (linear interpolation): for probability p and n
#' observations, the quantile is computed as:
#'
#' - h = (n-1) * p
#' - Interpolate between floor(h) and ceiling(h)
#'
#' This matches the default behavior of [stats::quantile()].
#'
#' @export
#' @examples
#' x <- as_mlx(1:10)
#' mlx_quantile(x, 0.5)  # median
#' mlx_quantile(x, c(0.25, 0.5, 0.75))  # quartiles
#'
#' # S3 method:
#' quantile(x, probs = c(0, 0.25, 0.5, 0.75, 1))
#'
#' # With axis parameter, quantile dimension replaces the reduced axis:
#' mat <- as_mlx(matrix(1:12, 3, 4))  # shape (3, 4)
#' result <- mlx_quantile(mat, c(0.25, 0.75), axis = 1)  # shape (2, 4)
#' result <- mlx_quantile(mat, 0.5, axis = 1)  # shape (1, 4)
#' result <- mlx_quantile(mat, 0.5, axis = 1, drop = TRUE)  # shape (4,)
mlx_quantile <- function(x, probs, axis = NULL, drop = FALSE, device = mlx_default_device()) {
  x <- as_mlx(x, device = device)

  # Validate probs
  if (!is.numeric(probs) || any(is.na(probs))) {
    stop("probs must be numeric without NA values", call. = FALSE)
  }
  if (any(probs < 0) || any(probs > 1)) {
    stop("probs must be in [0, 1]", call. = FALSE)
  }

  # Handle axis parameter
  if (!is.null(axis)) {
    if (length(axis) == 1) {
      # Single axis case
      axis <- as.integer(axis)
      axis_idx <- .mlx_normalize_axis_single(axis, x)
      sorted_x <- mlx_sort(x, axis = axis)  # mlx_sort uses 1-indexed already
      shape <- cpp_mlx_shape(x$ptr)
      n <- shape[axis]
    } else {
      # Multiple axes: flatten those axes and compute quantiles
      stop("Multiple axes not yet implemented for mlx_quantile", call. = FALSE)
    }
  } else {
    # No axis: flatten and sort entire array
    sorted_x <- mlx_sort(mlx_flatten(x))
    n <- length(x)
    axis <- NULL
  }

  if (n == 0) {
    stop("Cannot compute quantiles of empty array", call. = FALSE)
  }

  # Convert probs to mlx array
  probs_mlx <- as_mlx(probs, device = device)

  # Compute positions using type 7 formula: h = (n-1) * p
  # In 0-indexed: positions range from 0 to n-1
  if (n == 1) {
    # Special case: single element, return it for all probs
    if (is.null(axis)) {
      return(mlx_broadcast_to(sorted_x, length(probs)))
    } else {
      # For axis case, squeeze out the axis dimension and add quantile dimension
      target_shape <- dim(x)
      target_shape[axis] <- length(probs)
      return(mlx_broadcast_to(sorted_x, target_shape))
    }
  }

  h <- (n - 1) * probs_mlx

  # Get lower and upper indices (0-indexed for internal use)
  lower_idx <- floor(h)
  upper_idx <- mlx_clip(lower_idx + 1, 0, n - 1)  # Don't exceed array bounds

  # Compute interpolation weight
  weight <- h - lower_idx

  # Extract values at the indices
  if (is.null(axis)) {
    # Simple case: 1D sorted array, use direct indexing (1-indexed for R)
    lower_idx_1based <- lower_idx + 1
    upper_idx_1based <- upper_idx + 1
    lower_vals <- mlx_gather(sorted_x, list(lower_idx_1based), axes = 1L)
    upper_vals <- mlx_gather(sorted_x, list(upper_idx_1based), axes = 1L)
  } else {
    # Axis-specific case: use mlx_gather
    # mlx_gather accepts mlx arrays and expects 1-indexed indices
    lower_idx_1based <- lower_idx + 1
    upper_idx_1based <- upper_idx + 1

    lower_vals <- mlx_gather(sorted_x, list(lower_idx_1based), axes = axis)
    upper_vals <- mlx_gather(sorted_x, list(upper_idx_1based), axes = axis)

    # Reshape weight to be broadcastable
    # weight has shape (n_probs,), need to add dimensions for broadcasting
    # Result from gather has the axis dimension replaced by the index dimension
    # So we need to reshape weight to (n_probs, 1, 1, ...) with extra dims after axis
    lv_shape <- mlx_shape(lower_vals)
    weight_shape <- rep(1L, length(lv_shape))
    weight_shape[axis] <- length(probs)
    weight <- mlx_reshape(weight, weight_shape)
  }

  # Linear interpolation: (1 - weight) * lower + weight * upper
  result <- (1 - weight) * lower_vals + weight * upper_vals

  # Handle drop parameter
  if (!is.null(axis) && drop && length(probs) == 1) {
    # Remove the quantile dimension of size 1
    res_shape <- mlx_shape(result)
    new_dim <- res_shape[-axis]
    if (length(new_dim) == 0) {
      # Result is a scalar
      new_dim <- integer(0)
    }
    result <- mlx_reshape(result, new_dim)
  }

  return(result)
}

#' @rdname mlx_quantile
#' @export
#' @importFrom stats quantile
quantile.mlx <- function(x, probs, ...) {
  mlx_quantile(x, probs = probs, axis = NULL, device = x$device)
}
