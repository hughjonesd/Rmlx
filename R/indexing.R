#' Gather elements from an mlx array
#'
#' Wraps [`mlx.core.gather()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.gather)
#' so you can pull elements by axis. Provide one index per axis. Axes must
#' be positive integers (we don't allow negative indices, unlike Python).
#'
#' @inheritParams mlx_array_required
#' @param indices List of numeric/logical vectors or arrays (R or `mlx`). All
#'   entries must broadcast to a common shape.
#' @param axes Integer vector of axes (1-indexed). Defaults to the first
#'   `length(indices)` axes.
#' @return An `mlx` array containing the gathered elements.
#'
#' @section Element-wise indexing:
#' The output has the same shape as the indices. Each element of the output
#' is `x[index_1, index_2, ...]` from the corresponding position of each
#' index. See the examples below.
#'
#' @export
#' @examples
#' x <- as_mlx(matrix(1:9, 3, 3))
#'
#' # Simple cartesian gather:
#' as.matrix(mlx_gather(x, list(1:2, 1:2), axes = 1:2))
#'
#' # Element-wise pairs: grab a custom 2x2 grid of coordinates
#' row_idx <- matrix(c(1, 1,
#'                     2, 3), nrow = 2, byrow = TRUE)
#' col_idx <- matrix(c(1, 3,
#'                     2, 2), nrow = 2, byrow = TRUE)
#' as.array(mlx_gather(x, list(row_idx, col_idx), axes = c(1L, 2L)))
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
  if (any(is.na(axes))) {
    stop("`axes` must be numeric.", call. = FALSE)
  }
  if (anyDuplicated(axes)) {
    stop("`axes` must not contain duplicates.", call. = FALSE)
  }

  dims <- dim(x)
  ndim <- length(dims)
  if (any(axes < 1L | axes > ndim)) {
    stop("Each axis must fall within the array's dimensions. ",
         "Negative axes are not supported", call. = FALSE)
  }

  axes0 <- axes - 1L
  idx_dims <- lapply(idx_list, dim)
  normalized <- Map(function(idx, axis_len) {
    norm <- .normalize_index_vector(idx, axis_len)
    if (is.null(norm)) {
      stop("`indices` entries cannot be NULL.", call. = FALSE)
    }
    norm
  }, idx_list, dims[axes])

  use_take <- length(axes0) == 1L && length(idx_dims) == 1L &&
    (is.null(idx_dims[[1]]) || !length(idx_dims[[1]]))
  if (use_take) {
    ptr <- cpp_mlx_take(x$ptr, normalized[[1]], axes0[[1]])
    return(.mlx_wrap_result(ptr, x$device))
  }

  # Convert normalized vectors into mlx int64 arrays, reapplying the user
  # supplied shape when it still matches the number of elements.
  idx_mlx <- Map(function(vals, d) {
    if (!is.null(d) && length(d) && prod(d) == length(vals)) {
      dim(vals) <- d
    }
    as_mlx(vals, dtype = "int64", device = x$device)
  }, normalized, idx_dims)

  ptr <- cpp_mlx_gather(x$ptr, idx_mlx, axes0, x$device)
  res <- .mlx_wrap_result(ptr, x$device)

  res_dims <- dim(res)
  ndim <- length(dims)
  index_rank <- max(length(res_dims) - ndim, 0L)
  # Gather collapses indexed axes to length-1 trailing dims; rebuild the
  # original shape by keeping only the untouched axes after the index dims.
  keep_axes <- if (ndim) setdiff(seq_len(ndim), axes) else integer(0)
  kept <- if (length(keep_axes)) res_dims[index_rank + keep_axes] else integer(0)
  target <- c(res_dims[seq_len(index_rank)], kept)
  if (!length(target)) target <- integer(0)
  if (!identical(target, res_dims)) {
    res <- mlx_reshape(res, target)
  }
  res
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
  value <- as_mlx(value, dtype = mlx_dtype(x), device = x$device)

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
