#' Gather elements from an mlx array
#'
#' Mirrors [`mlx.core.gather()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.gather),
#' selecting elements according to index tensors along the specified axes. The
#' current implementation supports gathering along a single axis at a time.
#'
#' @inheritParams mlx_array_required
#' @param indices List of index tensors. Each element can be a numeric/logical
#'   vector, array, or an `mlx` array of integer type. Shapes must broadcast to a
#'   common result.
#' @param axes Integer vector of axes (1-indexed, negatives count from the end)
#'   corresponding to `indices`. Defaults to the first `length(indices)` axes.
#' @return An `mlx` array containing the gathered elements.
#' @export
#' @examples
#' x <- as_mlx(matrix(1:9, 3, 3))
#' idx_rows <- c(1L, 3L)
#' gathered <- mlx_gather(x, list(idx_rows), axes = 1L)
#' as.matrix(gathered)
mlx_gather <- function(x, indices, axes = NULL) {
  x <- as_mlx(x)

  if (!length(indices)) {
    stop("`indices` must contain at least one tensor.", call. = FALSE)
  }

  idx_list <- if (is.list(indices) && !is.mlx(indices)) {
    indices
  } else {
    list(indices)
  }

  if (is.null(axes)) {
    axes <- seq_along(idx_list)
  }
  axes <- as.integer(axes)
  if (length(axes) != length(idx_list)) {
    stop("`axes` must have the same length as `indices`.", call. = FALSE)
  }
  axes0 <- vapply(axes, function(ax) {
    if (ax < 0L) {
      length(dim(x)) + ax
    } else {
      ax - 1L
    }
  }, integer(1))
  if (length(axes0) != 1L) {
    stop("mlx_gather() currently supports gathering along a single axis.", call. = FALSE)
  }

  axis0 <- axes0[[1]]
  axis_len <- dim(x)[axis0 + 1L]
  idx_vals <- idx_list[[1]]
  idx_vec <- if (is.null(idx_vals)) NULL else as.vector(idx_vals)
  sel <- .normalize_index_vector(idx_vec, axis_len)
  ptr <- cpp_mlx_take(x$ptr, sel, axis0)
  .mlx_wrap_result(ptr, x$device)
}

#' Update a slice of an mlx array
#'
#' Wrapper around [`mlx.core.slice_update()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.slice_update)
#' that replaces a contiguous strided region with `value`.
#'
#' @inheritParams mlx_array_required
#' @param value Replacement `mlx` (or coercible) array. Must broadcast to the
#'   slice determined by `start`, `stop`, and `strides`.
#' @param start Integer vector (1-indexed) giving the inclusive starting index
#'   for each axis.
#' @param stop Integer vector (1-indexed) giving the inclusive stopping index
#'   for each axis.
#' @param strides Optional integer vector of strides (defaults to ones).
#' @return An `mlx` array with the specified slice replaced.
#' @export
#' @examples
#' x <- as_mlx(matrix(1:9, 3, 3))
#' replacement <- as_mlx(matrix(100:103, nrow = 2))
#' updated <- mlx_slice_update(x, replacement, start = c(1L, 2L), stop = c(2L, 3L))
#' as.matrix(updated)
mlx_slice_update <- function(x,
                             value,
                             start,
                             stop,
                             strides = NULL) {
  x <- as_mlx(x)
  value <- as_mlx(value, dtype = x$dtype, device = x$device)

  start <- as.integer(start)
  stop <- as.integer(stop)
  if (is.null(strides)) {
    strides <- rep.int(1L, length(start))
  }
  strides <- as.integer(strides)

  if (!(length(start) == length(stop) && length(stop) == length(strides))) {
    stop("`start`, `stop`, and `strides` must have the same length.", call. = FALSE)
  }

  if (any(start < 1L)) {
    stop("`start` must use 1-based indices (>= 1).", call. = FALSE)
  }
  if (any(stop < start)) {
    stop("Each `stop` entry must be >= the corresponding `start` value.", call. = FALSE)
  }

  start0 <- start - 1L
  stop0 <- stop

  ptr <- cpp_mlx_slice_update(x$ptr, value$ptr, start0, stop0, strides)
  .mlx_wrap_result(ptr, x$device)
}

# Internal helper for scatter-based updates on flattened tensors
#' Scatter helper used for `[<-` fallback paths.
#'
#' @param x Source `mlx` array.
#' @param indices Integer `mlx` array of flattened indices.
#' @param updates Replacement values as an `mlx` array.
#' @param axis Integer axis (0-indexed) supplied to MLX `scatter`.
#' @return An `mlx` array with the specified updates applied.
#' @noRd
.mlx_scatter_axis <- function(x, indices, updates, axis = 0L) {
  if (!is.mlx(x) || !is.mlx(indices) || !is.mlx(updates)) {
    stop("All inputs to .mlx_scatter_axis must be mlx arrays.", call. = FALSE)
  }
  ptr <- cpp_mlx_scatter(x$ptr, indices$ptr, updates$ptr, as.integer(axis))
  .mlx_wrap_result(ptr, x$device)
}
