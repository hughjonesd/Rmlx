#' @title Tensor shape helpers
#' @name mlx_shape_helpers
NULL

# Internal casting helper ----------------------------------------------------

.mlx_cast <- function(x, dtype = x$dtype, device = x$device) {
  if (!inherits(x, "mlx")) {
    stop("Expected an `mlx` tensor.", call. = FALSE)
  }
  if (identical(dtype, x$dtype) && identical(device, x$device)) {
    return(x)
  }
  ptr <- cpp_mlx_cast(x$ptr, dtype, device)
  new_mlx(ptr, x$dim, dtype, device)
}

.mlx_normalize_new_axes <- function(axes, dims) {
  if (length(axes) == 0L) {
    stop("axes must contain at least one element.", call. = FALSE)
  }
  axes <- as.integer(axes)
  if (any(is.na(axes))) {
    stop("axes cannot contain NA values.", call. = FALSE)
  }
  result_ndim <- length(dims) + 1L
  axes0 <- integer(length(axes))
  for (i in seq_along(axes)) {
    ax <- axes[[i]]
    if (ax < 0L) {
      ax <- ax + result_ndim
    } else {
      ax <- ax - 1L
    }
    if (ax < 0L || ax >= result_ndim) {
      stop("axis values are out of bounds.", call. = FALSE)
    }
    axes0[[i]] <- ax
  }
  sort(unique(axes0))
}

.mlx_normalize_new_axis <- function(axis, dims) {
  axes0 <- .mlx_normalize_new_axes(axis, dims)
  if (length(axes0) != 1L) {
    stop("axis must be a single integer.", call. = FALSE)
  }
  axes0
}

#' Stack MLX tensors along a new axis
#'
#' @param ... One or more tensors (or a single list of tensors) coercible to `mlx`.
#' @param axis Position of the new axis (1-indexed, negative values count from the end).
#' @return An `mlx` tensor with one additional dimension.
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' y <- as_mlx(matrix(5:8, 2, 2))
#' stacked <- mlx_stack(x, y, axis = 1)
mlx_stack <- function(..., axis = 1L) {
  tensors <- list(...)
  if (length(tensors) == 1L && is.list(tensors[[1]]) && !is.mlx(tensors[[1]])) {
    tensors <- tensors[[1]]
  }
  if (!length(tensors)) {
    stop("No tensors supplied.", call. = FALSE)
  }
  tensors <- lapply(tensors, function(x) if (is.mlx(x)) x else as_mlx(x))
  dtype <- Reduce(.promote_dtype, lapply(tensors, `[[`, "dtype"))
  device <- Reduce(.common_device, lapply(tensors, `[[`, "device"))
  tensors <- lapply(tensors, .mlx_cast, dtype = dtype, device = device)

  axis0 <- .mlx_normalize_new_axis(axis, tensors[[1]]$dim)
  ptr <- cpp_mlx_stack(tensors, axis0, device)
  .mlx_wrap_result(ptr, device)
}

#' Remove singleton dimensions
#'
#' @param x An `mlx` tensor.
#' @param axis Optional integer vector of axes (1-indexed) to remove. When `NULL`
#'   all axes of length one are removed.
#' @return An `mlx` tensor with the selected axes removed.
#' @export
#' @examples
#' x <- as_mlx(array(1:4, dim = c(1, 2, 2, 1)))
#' mlx_squeeze(x)
#' mlx_squeeze(x, axis = 1)
mlx_squeeze <- function(x, axis = NULL) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  if (is.null(axis)) {
    ptr <- cpp_mlx_squeeze(x$ptr, NULL)
  } else {
    axes <- .mlx_normalize_axes(axis, x)
    ptr <- cpp_mlx_squeeze(x$ptr, axes)
  }
  .mlx_wrap_result(ptr, x$device)
}

#' Insert singleton dimensions
#'
#' @param x An `mlx` tensor.
#' @param axis Integer vector of axis positions (1-indexed) where new singleton
#'   dimensions should be inserted.
#' @return An `mlx` tensor with additional dimensions of length one.
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_expand_dims(x, axis = 1)
mlx_expand_dims <- function(x, axis) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  axis0 <- .mlx_normalize_new_axes(axis, x$dim)
  ptr <- cpp_mlx_expand_dims(x$ptr, axis0)
  .mlx_wrap_result(ptr, x$device)
}

#' Repeat tensor elements
#'
#' @param x An `mlx` tensor.
#' @param repeats Number of repetitions.
#' @param axis Optional axis along which to repeat. When `NULL`, the tensor is
#'   flattened before repetition (matching NumPy semantics).
#' @return An `mlx` tensor with repeated values.
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_repeat(x, repeats = 2, axis = 2)
mlx_repeat <- function(x, repeats, axis = NULL) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  repeats <- as.integer(repeats)
  if (length(repeats) != 1L || repeats <= 0L || is.na(repeats)) {
    stop("repeats must be a positive integer.", call. = FALSE)
  }

  if (is.null(axis)) {
    ptr <- cpp_mlx_repeat(x$ptr, repeats, NULL)
  } else {
    axis0 <- .mlx_normalize_axis(axis, x)
    ptr <- cpp_mlx_repeat(x$ptr, repeats, axis0)
  }
  .mlx_wrap_result(ptr, x$device)
}

#' Tile a tensor
#'
#' @param x An `mlx` tensor.
#' @param reps Integer vector giving the number of repetitions for each axis.
#' @return An `mlx` tensor with tiled content.
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_tile(x, reps = c(1, 2))
mlx_tile <- function(x, reps) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  reps <- as.integer(reps)
  if (length(reps) == 0L || any(is.na(reps)) || any(reps <= 0L)) {
    stop("reps must be positive integers.", call. = FALSE)
  }
  ptr <- cpp_mlx_tile(x$ptr, reps)
  .mlx_wrap_result(ptr, x$device)
}

#' Roll tensor elements
#'
#' @param x An `mlx` tensor.
#' @param shift Integer vector giving the number of places by which elements are shifted.
#' @param axis Optional axis (or axes) along which elements are shifted.
#'   When `NULL`, the tensor is flattened and shifted.
#' @return An `mlx` tensor with elements circularly shifted.
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_roll(x, shift = 1, axis = 2)
mlx_roll <- function(x, shift, axis = NULL) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  shift <- as.integer(shift)
  if (length(shift) == 0L || any(is.na(shift))) {
    stop("shift must contain integer values.", call. = FALSE)
  }

  if (is.null(axis)) {
    ptr <- cpp_mlx_roll(x$ptr, shift, NULL)
  } else {
    axes0 <- .mlx_normalize_axes(axis, x)
    if (length(shift) != length(axes0)) {
      stop("shift and axis must have the same length.", call. = FALSE)
    }
    ptr <- cpp_mlx_roll(x$ptr, shift, axes0)
  }
  .mlx_wrap_result(ptr, x$device)
}

#' Elementwise conditional selection
#'
#' @param condition Logical `mlx` tensor (non-zero values are treated as `TRUE`).
#' @param x,y Tensors broadcastable to the shape of `condition`.
#' @return An `mlx` tensor where elements are drawn from `x` when
#'   `condition` is `TRUE`, otherwise from `y`.
#' @details Behaves like [ifelse()] for tensors, but evaluates both branches.
#' @export
#' @examples
#' cond <- as_mlx(matrix(c(TRUE, FALSE, TRUE, FALSE), 2, 2))
#' a <- as_mlx(matrix(1:4, 2, 2))
#' b <- as_mlx(matrix(5:8, 2, 2))
#' mlx_where(cond, a, b)
mlx_where <- function(condition, x, y) {
  condition <- if (is.mlx(condition)) condition else as_mlx(condition)
  x <- if (is.mlx(x)) x else as_mlx(x)
  y <- if (is.mlx(y)) y else as_mlx(y)

  result_dtype <- .promote_dtype(x$dtype, y$dtype)
  result_device <- .common_device(x$device, y$device)

  condition <- .mlx_cast(condition, dtype = "bool", device = result_device)
  x <- .mlx_cast(x, dtype = result_dtype, device = result_device)
  y <- .mlx_cast(y, dtype = result_dtype, device = result_device)

  ptr <- cpp_mlx_where(condition$ptr, x$ptr, y$ptr, result_dtype, result_device)
  .mlx_wrap_result(ptr, result_device)
}
