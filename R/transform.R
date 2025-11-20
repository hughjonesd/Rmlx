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
  new_mlx(ptr, x$device)
}

#' Argmax and argmin on mlx arrays
#'
#' @inheritParams common_params
#'
#' @details When `axis = NULL`, the array is flattened before computing extrema.
#' Setting `drop = FALSE` retains the reduced axis as length one in the
#' returned indices.
#'
#' @return An mlx array of indices. Indices are 1-based to match R's
#'   conventions.
#' @seealso
#'   [mlx.core.argmax](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.argmax),
#'   [mlx.core.argmin](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.argmin)
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
  new_mlx(ptr, x$device)
}

#' @rdname mlx_argmax
#' @export
mlx_argmin <- function(x, axis = NULL, drop = TRUE) {
  x <- as_mlx(x)
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_argmin(x$ptr, axis_idx, !isTRUE(drop))
  new_mlx(ptr, x$device)
}

#' Sort and argsort for mlx arrays
#'
#' `mlx_sort()` returns sorted values along the specified axis. `mlx_argsort()`
#' returns the indices that would sort the array.
#'
#' @inheritParams mlx_argmax
#'
#' @return An mlx array containing sorted values (for `mlx_sort()`) or
#'   **1-based indices** (for `mlx_argsort()`). The indices follow R's indexing
#'   convention and can be used directly with R's `[` operator.
#' @details
#' `mlx_argsort()` returns **1-based indices** that would sort the array in
#' ascending order. This follows R's indexing convention (unlike the underlying
#' MLX library which uses 0-based indexing). The returned indices can be used
#' directly to reorder the original array.
#'
#' For partial sorting (finding elements up to a certain rank without fully
#' sorting), see [mlx_partition()] and [mlx_argpartition()].
#'
#' @seealso
#'   [mlx.core.sort](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.sort),
#'   [mlx.core.argsort](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.argsort)
#' @export
#' @examples
#' x <- as_mlx(c(3, 1, 4, 2))
#' mlx_sort(x)
#'
#' # Returns 1-based indices
#' idx <- mlx_argsort(x)
#' as.integer(as.matrix(idx))  # [1] 2 4 1 3
#'
#' # Can be used directly with R indexing
#' original <- c(3, 1, 4, 2)
#' sorted_idx <- as.integer(as.matrix(mlx_argsort(as_mlx(original))))
#' original[sorted_idx]  # [1] 1 2 3 4
#'
#' mlx_sort(mlx_matrix(1:6, 2, 3), axis = 1)
mlx_sort <- function(x, axis = NULL) {
  x <- as_mlx(x)
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_sort(x$ptr, axis_idx)
  new_mlx(ptr, x$device)
}

#' @rdname mlx_sort
#' @export
mlx_argsort <- function(x, axis = NULL) {
  x <- as_mlx(x)
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_argsort(x$ptr, axis_idx)
  new_mlx(ptr, x$device)
}

#' Top-k selection and partitioning on mlx arrays
#'
#' `mlx_topk()` returns the largest `k` values. `mlx_partition()` and
#' `mlx_argpartition()` perform partial sorting, rearranging elements so that
#' the element at position `kth` is in its correctly sorted position, with all
#' smaller elements before it and all larger elements after it. This is more
#' efficient than full sorting when you only need elements up to a certain rank.
#'
#' @inheritParams mlx_argmax
#' @param k Positive integer specifying the number of elements to select.
#' @param kth Zero-based index of the element that should be placed in-order
#'   after partitioning.
#'
#' @return An mlx array. For `mlx_argpartition()`, returns 1-based indices
#'   (following R conventions) showing the partition ordering.
#' @details
#' - `mlx_topk()` returns the largest `k` values along the specified axis.
#' - `mlx_partition()` rearranges elements so the kth element is correctly positioned.
#' - `mlx_argpartition()` returns the **1-based indices** that would partition
#'   the array. This follows R's indexing convention (unlike the underlying MLX
#'   library which uses 0-based indexing).
#'
#' Use `mlx_argsort()` if you need fully sorted indices.
#'
#' @seealso
#'   [mlx.core.topk](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.topk),
#'   [mlx.core.partition](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.partition),
#'   [mlx.core.argpartition](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.argpartition)
#' @export
#' @examples
#' scores <- as_mlx(c(0.7, 0.2, 0.9, 0.4))
#' mlx_topk(scores, k = 2)
#' mlx_partition(scores, kth = 1)
#'
#' # Returns 1-based indices
#' idx <- mlx_argpartition(scores, kth = 1)
#' as.integer(as.matrix(idx))  # 1-based indices
#'
#' mlx_topk(mlx_matrix(1:6, 2, 3), k = 1, axis = 1)
mlx_topk <- function(x, k, axis = NULL) {
  x <- as_mlx(x)
  if (length(k) != 1L || !is.finite(k) || k <= 0) {
    stop("k must be a positive finite scalar.", call. = FALSE)
  }
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_topk(x$ptr, as.integer(k), axis_idx)
  new_mlx(ptr, x$device)
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
  new_mlx(ptr, x$device)
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
  new_mlx(ptr, x$device)
}

#' Normalize multiple axes to 0-indexed values
#'
#' @param axes Integer vector of 1-indexed axes or `NULL`.
#' @param x mlx array providing dimensionality information.
#' @return Integer vector of 0-indexed axes or `NULL`.
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

#' Normalize a single axis to 0-indexed form
#'
#' @param axis Single 1-indexed axis value.
#' @param x mlx array providing dimensionality information.
#' @return Integer scalar (0-indexed).
#' @noRd
.mlx_normalize_axis_single <- function(axis, x) {
  if (is.na(axis)) {
    stop("axis cannot be NA", call. = FALSE)
  }
  ndim <- length(dim(x))
  if (axis < 1L || axis > ndim) {
    stop(sprintf("axis=%d is out of bounds for an array with %d dimensions.", axis, ndim), call. = FALSE)
  }
  axis - 1L
}

#' Normalize a possibly `NULL` axis
#'
#' @param axis Single 1-indexed axis or `NULL`.
#' @param x mlx array providing dimensionality information.
#' @return Integer scalar (0-indexed) or `NULL`.
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
#' @inheritParams common_params
#' @param drop Logical indicating whether the reduced axes should be dropped (default `TRUE`).
#' @return An mlx array containing log-sum-exp results.
#' @seealso [mlx.core.logsumexp](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.logsumexp)
#' @export
#' @examples
#' x <- mlx_matrix(1:6, 2, 3)
#' mlx_logsumexp(x)
#' mlx_logsumexp(x, axes = 2)
mlx_logsumexp <- function(x, axes = NULL, drop = TRUE) {
  x <- as_mlx(x)
  axes_idx <- .mlx_normalize_axes(axes, x)
  ptr <- cpp_mlx_logsumexp(x$ptr, axes_idx, !isTRUE(drop))
  new_mlx(ptr, x$device)
}

#' Log cumulative sum exponential for mlx arrays
#'
#' @inheritParams common_params
#' @param axis Optional axis (single integer) to operate over.
#' @param reverse Logical flag for reverse accumulation.
#' @param inclusive Logical flag controlling inclusivity.
#' @return An mlx array.
#' @seealso [mlx.core.logaddexp](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.logaddexp)
#' @export
#' @examples
#' x <- as_mlx(1:4)
#' mlx_logcumsumexp(x)
#' m <- mlx_matrix(1:6, 2, 3)
#' mlx_logcumsumexp(m, axis = 2)
mlx_logcumsumexp <- function(x, axis = NULL, reverse = FALSE, inclusive = TRUE) {
  x <- as_mlx(x)
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_logcumsumexp(x$ptr, axis_idx, reverse, inclusive)
  new_mlx(ptr, x$device)
}

#' Softmax for mlx arrays
#'
#' @inheritParams common_params
#' @param precise Logical; compute in higher precision for stability.
#' @return An mlx array with normalized probabilities.
#' @seealso [mlx.core.softmax](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.softmax)
#' @export
#' @examples
#' x <- mlx_matrix(1:6, 2, 3)
#' sm <- mlx_softmax(x, axes = 2)
#' rowSums(sm)
mlx_softmax <- function(x, axes = NULL, precise = FALSE) {
  x <- as_mlx(x)
  axes_idx <- .mlx_normalize_axes(axes, x)
  ptr <- cpp_mlx_softmax(x$ptr, axes_idx, precise)
  new_mlx(ptr, x$device)
}
