#' Argmax and argmin on MLX tensors
#'
#' @param x An object coercible to `mlx`.
#' @param axis Optional axis to operate over (1-indexed like R). When `NULL`, the
#'   tensor is flattened first.
#' @param keepdims Logical; retain reduced dimensions with length one.
#'
#' @return An `mlx` tensor of indices.
#' @export
mlx_argmax <- function(x, axis = NULL, keepdims = FALSE) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_argmax(x$ptr, axis_idx, keepdims)
  dim <- cpp_mlx_shape(ptr)
  dtype <- cpp_mlx_dtype(ptr)
  new_dim <- if (length(dim)) as.integer(dim) else integer(0)
  new_mlx(ptr, new_dim, dtype, x$device)
}

#' @rdname mlx_argmax
#' @export
mlx_argmin <- function(x, axis = NULL, keepdims = FALSE) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_argmin(x$ptr, axis_idx, keepdims)
  dim <- cpp_mlx_shape(ptr)
  dtype <- cpp_mlx_dtype(ptr)
  new_dim <- if (length(dim)) as.integer(dim) else integer(0)
  new_mlx(ptr, new_dim, dtype, x$device)
}

#' Sort and argsort for MLX tensors
#'
#' @inheritParams mlx_argmax
#'
#' @return An `mlx` tensor containing sorted values (for `mlx_sort`) or indices
#'   (for `mlx_argsort`).
#' @details Indices returned by `mlx_argsort()` and `mlx_argpartition()` use
#'   zero-based offsets, matching MLX's native conventions.
#' @export
mlx_sort <- function(x, axis = NULL) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_sort(x$ptr, axis_idx)
  dim <- cpp_mlx_shape(ptr)
  dtype <- cpp_mlx_dtype(ptr)
  new_dim <- if (length(dim)) as.integer(dim) else integer(0)
  new_mlx(ptr, new_dim, dtype, x$device)
}

#' @rdname mlx_sort
#' @export
mlx_argsort <- function(x, axis = NULL) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_argsort(x$ptr, axis_idx)
  dim <- cpp_mlx_shape(ptr)
  dtype <- cpp_mlx_dtype(ptr)
  new_dim <- if (length(dim)) as.integer(dim) else integer(0)
  new_mlx(ptr, new_dim, dtype, x$device)
}

#' Top-k selection and partitioning on MLX tensors
#'
#' @inheritParams mlx_argmax
#' @param k Positive integer specifying the number of elements to select.
#' @param kth Zero-based index of the element that should be placed in-order
#'   after partitioning.
#'
#' @return An `mlx` tensor.
#' @details `mlx_topk()` returns the largest `k` values as reported by MLX. Use
#'   `mlx_argsort()` if you need the associated indices.
#' @export
mlx_topk <- function(x, k, axis = NULL) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  if (length(k) != 1L || !is.finite(k) || k <= 0) {
    stop("k must be a positive finite scalar.", call. = FALSE)
  }
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_topk(x$ptr, as.integer(k), axis_idx)
  dim <- cpp_mlx_shape(ptr)
  dtype <- cpp_mlx_dtype(ptr)
  new_dim <- if (length(dim)) as.integer(dim) else integer(0)
  new_mlx(ptr, new_dim, dtype, x$device)
}

#' @rdname mlx_topk
#' @export
mlx_partition <- function(x, kth, axis = NULL) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  if (length(kth) != 1L || !is.finite(kth) || kth < 0) {
    stop("kth must be a non-negative finite scalar.", call. = FALSE)
  }
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_partition(x$ptr, as.integer(kth), axis_idx)
  dim <- cpp_mlx_shape(ptr)
  dtype <- cpp_mlx_dtype(ptr)
  new_dim <- if (length(dim)) as.integer(dim) else integer(0)
  new_mlx(ptr, new_dim, dtype, x$device)
}

#' @rdname mlx_topk
#' @export
mlx_argpartition <- function(x, kth, axis = NULL) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  if (length(kth) != 1L || !is.finite(kth) || kth < 0) {
    stop("kth must be a non-negative finite scalar.", call. = FALSE)
  }
  axis_idx <- .mlx_normalize_axis(axis, x)
  ptr <- cpp_mlx_argpartition(x$ptr, as.integer(kth), axis_idx)
  dim <- cpp_mlx_shape(ptr)
  dtype <- cpp_mlx_dtype(ptr)
  new_dim <- if (length(dim)) as.integer(dim) else integer(0)
  new_mlx(ptr, new_dim, dtype, x$device)
}

# Internal helper: convert R axis (1-indexed) to zero-based for C++.
.mlx_normalize_axis <- function(axis, x) {
  if (is.null(axis)) {
    return(NULL)
  }
  if (length(axis) != 1L) {
    stop("axis must be NULL or a single integer.", call. = FALSE)
  }
  ndim <- length(x$dim)
  axis <- as.integer(axis)
  if (axis < 0L) {
    axis <- ndim + axis + 1L
  }
  if (axis < 1L || axis > ndim) {
    stop(sprintf("axis=%d is out of bounds for an array with %d dimensions.", axis, ndim), call. = FALSE)
  }
  axis - 1L
}
