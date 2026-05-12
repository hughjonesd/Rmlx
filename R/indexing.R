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
#' The output has the same shape as the indices (after broadcasting). Each element
#' `[i, j, ...]`of the output
#' is `x[index_1[i, j, ...], index_2[i, j, ...], ...]` from the corresponding
#' position of each index. See the examples below.
#'
#' @export
#' @examples
#' x <- mlx_matrix(1:9, 3, 3)
#'
#' # Simple cartesian gather:
#' mlx_gather(x, list(1:2, 1:2))
#'
#' # Element-wise pairs: grab a custom 2x2 grid of coordinates
#' row_idx <- matrix(c(1, 1,
#'                     2, 3), nrow = 2, byrow = TRUE)
#' col_idx <- matrix(c(1, 3,
#'                     2, 2), nrow = 2, byrow = TRUE)
#'
#' # A 2x2 matrix where (e.g.) the bottom right element is x[3, 2]
#' mlx_gather(x, list(row_idx, col_idx))
mlx_gather <- function(x, indices, axes = NULL) {
  x <- as_mlx(x)

  if (! length(indices)) {
    stop("`indices` must contain at least one tensor.", call. = FALSE)
  }

  idx_list <- if (is.list(indices) && !is_mlx(indices)) {
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
  if (anyNA(axes)) {
    stop("`axes` must be numeric.", call. = FALSE)
  }
  if (anyDuplicated(axes)) {
    stop("`axes` must not contain duplicates.", call. = FALSE)
  }

  shape <- mlx_shape(x)
  ndim <- length(shape)
  if (any(axes < 1L | axes > ndim)) {
    stop("Each axis must fall within the array's dimensions. ",
         "Negative axes are not supported", call. = FALSE)
  }

  axes0 <- axes - 1L
  idx_dims <- lapply(idx_list, dim)
  normalized <- Map(function(idx, axis_len) {
    norm <- normalize_index(idx, axis_len, assign = FALSE, allow_dims = TRUE)
    norm - 1L
  }, idx_list, shape[axes])

  use_take <- length(axes0) == 1L && length(idx_dims) == 1L &&
    (is.null(idx_dims[[1]]) || !length(idx_dims[[1]]))
  if (use_take) {
    ptr <- cpp_mlx_take(x$ptr, as.vector(normalized[[1]]), axes0[[1]])
    return(new_mlx(ptr))
  }

  # Convert normalized vectors into mlx int32 arrays, reapplying the user
  # supplied shape when it still matches the number of elements.
  idx_mlx <- Map(function(vals, d) {
    if (!is.null(d) && length(d) > 0L && prod(d) == length(vals)) {
      dim(vals) <- d
    }
    as_mlx(vals, dtype = "int32")
  }, normalized, idx_dims)

  ptr <- cpp_mlx_gather(x$ptr, idx_mlx, axes0)
  res <- new_mlx(ptr)

  res_dims <- mlx_shape(res)
  ndim <- length(shape)
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

.mlx_index_array <- function(indices, axis_len) {
  if (!is.numeric(indices) && !is_mlx(indices)) {
    stop("indices must be numeric or an mlx array.", call. = FALSE)
  }
  if (!is_mlx(indices) && anyNA(indices)) {
    stop("indices cannot contain NA values.", call. = FALSE)
  }

  idx_mlx <- as_mlx(indices)
  if (identical(mlx_dtype(idx_mlx), "bool")) {
    stop("indices must be integer positions, not booleans.", call. = FALSE)
  }
  if (any(idx_mlx != floor(idx_mlx))) {
    stop("indices must be whole numbers.", call. = FALSE)
  }
  if (any(idx_mlx <= 0)) {
    stop("indices must be positive and 1-based.", call. = FALSE)
  }
  if (any(idx_mlx > axis_len)) {
    stop("indices are out of bounds for the selected axis.", call. = FALSE)
  }

  mlx_cast(idx_mlx - 1L, dtype = "int32")
}

#' Take values using per-position axis indices
#'
#' Mirrors [`mlx.core.take_along_axis()`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take_along_axis.html)
#' while accepting 1-based R indices.
#'
#' @inheritParams mlx_array_required
#' @param indices Integer positions along `axis`. Must be broadcast-compatible
#'   with `x` except at the selected axis.
#' @param axis Axis to index (1-based).
#' @return An `mlx` array.
#' @export
#' @examples
#' x <- mlx_matrix(1:12, nrow = 3, ncol = 4)
#' idx <- matrix(c(1L, 4L,
#'                 2L, 3L,
#'                 4L, 1L), nrow = 3, byrow = TRUE)
#' mlx_take_along_axis(x, idx, axis = 2L)
mlx_take_along_axis <- function(x, indices, axis) {
  x <- as_mlx(x)
  axis_idx <- normalize_axis_single(as.integer(axis), x)
  idx_mlx <- .mlx_index_array(indices, dim(x)[axis])
  ptr <- cpp_mlx_take_along_axis(x$ptr, idx_mlx$ptr, axis_idx)
  new_mlx(ptr)
}

#' Write values using per-position axis indices
#'
#' Mirrors [`mlx.core.put_along_axis()`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.put_along_axis.html)
#' while accepting 1-based R indices.
#'
#' @inheritParams mlx_take_along_axis
#' @param values Replacement values.
#' @return An updated `mlx` array.
#' @export
#' @examples
#' x <- mlx_matrix(1:12, nrow = 3, ncol = 4)
#' idx <- matrix(c(1L, 4L,
#'                 2L, 3L,
#'                 4L, 1L), nrow = 3, byrow = TRUE)
#' values <- matrix(c(100, 200,
#'                    300, 400,
#'                    500, 600), nrow = 3, byrow = TRUE)
#' mlx_put_along_axis(x, idx, values, axis = 2L)
mlx_put_along_axis <- function(x, indices, values, axis) {
  x <- as_mlx(x)
  axis_idx <- normalize_axis_single(as.integer(axis), x)
  idx_mlx <- .mlx_index_array(indices, dim(x)[axis])
  values_mlx <- as_mlx(values, dtype = mlx_dtype(x))
  ptr <- cpp_mlx_put_along_axis(x$ptr, idx_mlx$ptr, values_mlx$ptr, axis_idx)
  new_mlx(ptr)
}

#' Add values using per-position axis indices
#'
#' Mirrors [`mlx.core.scatter_add_axis()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.scatter_add_axis)
#' while accepting 1-based R indices.
#'
#' @inheritParams mlx_put_along_axis
#' @return An updated `mlx` array after additive scatter.
#' @export
#' @examples
#' x <- mlx_matrix(1:12, nrow = 3, ncol = 4)
#' idx <- matrix(c(1L, 1L,
#'                 2L, 3L,
#'                 4L, 4L), nrow = 3, byrow = TRUE)
#' values <- matrix(c(10, 20,
#'                    30, 40,
#'                    50, 60), nrow = 3, byrow = TRUE)
#' mlx_scatter_add_axis(x, idx, values, axis = 2L)
mlx_scatter_add_axis <- function(x, indices, values, axis) {
  x <- as_mlx(x)
  axis_idx <- normalize_axis_single(as.integer(axis), x)
  idx_mlx <- .mlx_index_array(indices, dim(x)[axis])
  values_mlx <- as_mlx(values, dtype = mlx_dtype(x))
  ptr <- cpp_mlx_scatter_add_axis(x$ptr, idx_mlx$ptr, values_mlx$ptr, axis_idx)
  new_mlx(ptr)
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
#'
#' @section Difference from Python/C++:
#' Unlike Python's slice notation `array[start:stop]` which uses an exclusive upper bound,
#' `mlx_slice_update()` uses **inclusive** bounds for both `start` and `stop` to match
#' R conventions and to be consistent with [mlx_arange()] and [mlx_linspace()].
#'
#' @export
#' @examples
#' x <- mlx_matrix(1:9, 3, 3)
#' replacement <- mlx_matrix(100:103, nrow = 2)
#' updated <- mlx_slice_update(x, replacement, start = c(1L, 2L), stop = c(2L, 3L))
#' updated
mlx_slice_update <- function(x,
                             value,
                             start,
                             stop,
                             strides = NULL) {
  x <- as_mlx(x)
  value <- as_mlx(value, dtype = mlx_dtype(x))

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
  new_mlx(ptr)
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
.mlx_scatter_axis <- function(x, indices, updates, axes) {
  idx_list <- if (is.list(indices)) indices else list(indices)
  ptr <- cpp_mlx_scatter(x$ptr, idx_list, updates$ptr, as.integer(axes))
  new_mlx(ptr)
}
