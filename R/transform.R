#' Hadamard transform for MLX arrays
#'
#' Multiplies the last dimension of `x` by the Sylvester-Hadamard matrix of the
#' corresponding size. The transform expects the length of the last axis to be a
#' power of two.
#'
#' @inheritParams common_params
#' @param scale Optional numeric scalar applied to the result. MLX defaults to
#'   `1 / sqrt(n)` where `n` is the size of the transformed axis; set `scale`
#'   to override the factor (for example, `scale = 1` yields the unnormalised
#'   Hadamard transform).
#' @return An `mlx` array containing the Hadamard-transformed values.
#' @seealso <https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.hadamard_transform>
#' @export
#' @examples
#' x <- as_mlx(c(1, -1))
#' as.vector(mlx_hadamard_transform(x))
#' as.vector(mlx_hadamard_transform(x, scale = 1))
mlx_hadamard_transform <- function(x, scale = NULL) {
  x <- as_mlx(x)

  if (!is.null(scale)) {
    if (length(scale) != 1L || !is.numeric(scale) || anyNA(scale)) {
      stop("`scale` must be NULL or a single numeric value.", call. = FALSE)
    }
    scale <- as.numeric(scale)
  }

  ptr <- cpp_mlx_hadamard_transform(x$ptr, scale, x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' Argmax and argmin on mlx arrays
#'
#' @inheritParams common_params
#' @param axis Optional axis to operate over (1-indexed like R). When `NULL`, the
#'   array is flattened first.
#' @param drop Logical; when `TRUE` (default) the reduced axis is removed.
#'   Set to `FALSE` to keep the axis as length one.
#'
#' @return An mlx array of indices. Indices are 1-based to match R's
#'   conventions.
#' @seealso
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.argmax},
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.argmin}
#' @export
#' @examples
#' x <- as_mlx(matrix(c(1, 5, 3, 2), 2, 2))
#' mlx_argmax(x)
#' mlx_argmax(x, axis = 1)
#' mlx_argmin(x)
mlx_argmax <- function(x, axis = NULL, drop = TRUE) {
  x <- as_mlx(x)
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_argmax(x$ptr, axis_idx, !isTRUE(drop))
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_argmax
#' @export
mlx_argmin <- function(x, axis = NULL, drop = TRUE) {
  x <- as_mlx(x)
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_argmin(x$ptr, axis_idx, !isTRUE(drop))
  .mlx_wrap_result(ptr, x$device)
}

#' Sort and argsort for mlx arrays
#'
#' @inheritParams mlx_argmax
#'
#' @return An mlx array containing sorted values (for `mlx_sort`) or indices
#'   (for `mlx_argsort`).
#' @details Indices returned by `mlx_argsort()` and `mlx_argpartition()` use
#'   zero-based offsets, matching MLX's native conventions.
#' @seealso
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.sort},
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.argsort}
#' @export
#' @examples
#' x <- as_mlx(c(3, 1, 4, 2))
#' mlx_sort(x)
#' mlx_argsort(x)
#' mlx_sort(as_mlx(matrix(1:6, 2, 3)), axis = 1)
mlx_sort <- function(x, axis = NULL) {
  x <- as_mlx(x)
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_sort(x$ptr, axis_idx)
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_sort
#' @export
mlx_argsort <- function(x, axis = NULL) {
  x <- as_mlx(x)
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_argsort(x$ptr, axis_idx)
  .mlx_wrap_result(ptr, x$device)
}

#' Top-k selection and partitioning on mlx arrays
#'
#' @inheritParams mlx_argmax
#' @param k Positive integer specifying the number of elements to select.
#' @param kth Zero-based index of the element that should be placed in-order
#'   after partitioning.
#'
#' @return An mlx array.
#' @details `mlx_topk()` returns the largest `k` values as reported by MLX. Use
#'   `mlx_argsort()` if you need the associated indices.
#' @seealso
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.topk},
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.partition},
#'   \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.argpartition}
#' @export
#' @examples
#' scores <- as_mlx(c(0.7, 0.2, 0.9, 0.4))
#' mlx_topk(scores, k = 2)
#' mlx_partition(scores, kth = 1)
#' mlx_argpartition(scores, kth = 1)
#' mlx_topk(as_mlx(matrix(1:6, 2, 3)), k = 1, axis = 1)
mlx_topk <- function(x, k, axis = NULL) {
  x <- as_mlx(x)
  if (length(k) != 1L || !is.finite(k) || k <= 0) {
    stop("k must be a positive finite scalar.", call. = FALSE)
  }
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_topk(x$ptr, as.integer(k), axis_idx)
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_topk
#' @export
mlx_partition <- function(x, kth, axis = NULL) {
  x <- as_mlx(x)
  if (length(kth) != 1L || !is.finite(kth) || kth < 0) {
    stop("kth must be a non-negative finite scalar.", call. = FALSE)
  }
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_partition(x$ptr, as.integer(kth), axis_idx)
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_topk
#' @export
mlx_argpartition <- function(x, kth, axis = NULL) {
  x <- as_mlx(x)
  if (length(kth) != 1L || !is.finite(kth) || kth < 0) {
    stop("kth must be a non-negative finite scalar.", call. = FALSE)
  }
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_argpartition(x$ptr, as.integer(kth), axis_idx)
  .mlx_wrap_result(ptr, x$device)
}

#' Normalize multiple axes to 0-indexed
#'
#' @param axes Integer vector of 1-indexed axes (negatives allowed) or NULL.
#' @param x mlx array providing dimensionality.
#' @return Integer vector (0-indexed) or NULL.
#' @noRd
.mlx_normalize_axes <- function(axes, x) {
  if (is.null(axes)) {
    return(NULL)
  }
  if (!length(axes)) {
    stop("axes must contain at least one element.", call. = FALSE)
  }
  axes <- as.integer(axes)
  vapply(axes, .mlx_normalize_axis_single, integer(1), x = x)
}

#' Convert single 1-indexed axis to 0-indexed
#'
#' @param axis Integer (1-indexed, negatives allowed).
#' @param x mlx array providing dimensionality.
#' @return Integer scalar (0-indexed).
#' @noRd
.mlx_normalize_axis_single <- function(axis, x) {
  if (is.na(axis)) {
    stop("axis cannot be NA", call. = FALSE)
  }
  ndim <- length(x$dim)
  if (axis < 0L) {
    axis <- ndim + axis + 1L
  }
  if (axis < 1L || axis > ndim) {
    stop(sprintf("axis=%d is out of bounds for an array with %d dimensions.", axis, ndim), call. = FALSE)
  }
  axis - 1L
}

#' Convert single axis to 0-indexed or return NULL
#'
#' @param axis Integer (1-indexed, negatives allowed) or NULL.
#' @param x mlx array providing dimensionality.
#' @return Integer scalar (0-indexed) or NULL.
#' @noRd
.mlx_normalize_axis <- function(axis, x) {
  if (is.null(axis)) {
    return(NULL)
  }
  if (length(axis) != 1L) {
    stop("axis must be NULL or a single integer.", call. = FALSE)
  }
  axis <- as.integer(axis)
  .mlx_normalize_axis_single(axis, x)
}

#' Log-sum-exp reduction for mlx arrays
#'
#' @inheritParams mlx_argmax
#' @param drop Logical indicating whether the reduced axes should be dropped (default `TRUE`).
#' @return An mlx array containing log-sum-exp results.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.logsumexp}
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 2, 3))
#' as.matrix(mlx_logsumexp(x))
#' as.matrix(mlx_logsumexp(x, axis = 2))
mlx_logsumexp <- function(x, axis = NULL, drop = TRUE) {
  x <- as_mlx(x)
  axes_idx <- .mlx_normalize_axes(axis, x)
  ptr <- cpp_mlx_logsumexp(x$ptr, axes_idx, !isTRUE(drop))
  .mlx_wrap_result(ptr, x$device)
}

#' Log cumulative sum exponential for mlx arrays
#'
#' @inheritParams mlx_argmax
#' @param axis Optional axis (single integer) to operate over.
#' @param reverse Logical flag for reverse accumulation.
#' @param inclusive Logical flag controlling inclusivity.
#' @return An mlx array.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.logaddexp}
#' @export
#' @examples
#' x <- as_mlx(1:4)
#' as.vector(as.matrix(mlx_logcumsumexp(x)))
#' m <- as_mlx(matrix(1:6, 2, 3))
#' as.matrix(mlx_logcumsumexp(m, axis = 2))
mlx_logcumsumexp <- function(x, axis = NULL, reverse = FALSE, inclusive = TRUE) {
  x <- as_mlx(x)
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_logcumsumexp(x$ptr, axis_idx, reverse, inclusive)
  .mlx_wrap_result(ptr, x$device)
}

#' Softmax for mlx arrays
#'
#' @inheritParams mlx_argmax
#' @param precise Logical; compute in higher precision for stability.
#' @return An mlx array with normalized probabilities.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.softmax}
#' @export
#' @examples
#' x <- as_mlx(matrix(c(1, 2, 3, 4, 5, 6), 2, 3))
#' sm <- mlx_softmax(x, axis = 2)
#' rowSums(as.matrix(sm))
mlx_softmax <- function(x, axis = NULL, precise = FALSE) {
  x <- as_mlx(x)
  axes_idx <- .mlx_normalize_axes(axis, x)
  ptr <- cpp_mlx_softmax(x$ptr, axes_idx, precise)
  .mlx_wrap_result(ptr, x$device)
}
