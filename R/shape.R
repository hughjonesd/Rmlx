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

#' Pad or split MLX tensors
#'
#' @description
#' * `mlx_pad()` mirrors the MLX padding primitive, enlarging each axis according
#'   to `pad_width`. Values are added symmetrically (`pad_width[i, 1]` before,
#'   `pad_width[i, 2]` after) using the specified `mode`.
#' * `mlx_split()` divides a tensor along an axis either into equal sections
#'   (`sections` scalar) or at explicit 1-based split points (`sections` vector),
#'   returning a list of `mlx` tensors.
#'
#' @param x An object coercible to `mlx` via [as_mlx()].
#' @param pad_width Padding extents. Supply a single integer, a length-two
#'   numeric vector, or a matrix/list with one `(before, after)` pair per padded
#'   axis.
#' @param value Constant fill value used when `mode = "constant"`.
#' @param mode Padding mode passed to MLX (e.g., `"constant"`, `"edge"`,
#'   `"reflect"`).
#' @param axes Optional integer vector of axes (1-indexed, negatives count from
#'   the end) to which `pad_width` applies. Unlisted axes receive zero padding.
#' @param sections Either a single integer (number of equal parts) or an integer
#'   vector of 1-based split points along `axis`.
#' @param axis Axis (1-indexed, negatives count from the end) to operate on.
#' @return For `mlx_pad()`, an `mlx` tensor; for `mlx_split()`, a list of `mlx`
#'   tensors.
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' padded <- mlx_pad(x, pad_width = 1)
#' padded_cols <- mlx_pad(x, pad_width = c(0, 1), axes = 2)
#' parts <- mlx_split(x, sections = 2, axis = 1)
#' custom_parts <- mlx_split(x, sections = c(1), axis = 2)
mlx_pad <- function(x,
                    pad_width,
                    value = 0,
                    mode = c("constant", "edge", "reflect", "symmetric"),
                    axes = NULL) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  mode <- match.arg(mode)

  ndim <- length(x$dim)
  if (ndim == 0L) {
    stop("Cannot pad a scalar mlx tensor.", call. = FALSE)
  }
  if (length(value) != 1L || !is.finite(value)) {
    stop("value must be a single finite numeric scalar.", call. = FALSE)
  }

  target_axes <- if (is.null(axes)) {
    seq_len(ndim)
  } else {
    axes <- as.integer(axes)
    if (!length(axes)) {
      stop("axes must contain at least one axis.", call. = FALSE)
    }
    vapply(axes, function(ax) .mlx_normalize_axis_single(ax, x) + 1L, integer(1))
  }

  pad_matrix <- matrix(0L, nrow = ndim, ncol = 2)
  pad_pairs <- .parse_pad_width(pad_width, length(target_axes))
  if (any(pad_pairs < 0L)) {
    stop("pad_width values must be non-negative.", call. = FALSE)
  }
  pad_matrix[target_axes, ] <- pad_pairs

  ptr <- cpp_mlx_pad(
    x$ptr,
    pad_matrix,
    as.numeric(value),
    x$dtype,
    x$device,
    mode
  )
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_pad
#' @export
mlx_split <- function(x, sections, axis = 1L) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  if (missing(sections)) {
    stop("sections must be supplied.", call. = FALSE)
  }
  axis_idx <- .mlx_normalize_axis(axis, x)
  dim_len <- x$dim[axis_idx + 1L]

  sections <- as.integer(sections)
  sections <- sections[!is.na(sections)]
  if (!length(sections)) {
    stop("sections must contain at least one integer.", call. = FALSE)
  }

  if (length(sections) == 1L) {
    num <- sections[[1]]
    if (num <= 0L || dim_len %% num != 0L) {
      stop("sections must evenly divide the axis length.", call. = FALSE)
    }
    ptrs <- cpp_mlx_split(
      x$ptr,
      num_splits_ = num,
      indices_ = NULL,
      axis = axis_idx,
      dtype_str = x$dtype,
      device_str = x$device
    )
  } else {
    if (any(sections <= 0L) || any(sections >= dim_len)) {
      stop("Split points must be between 1 and the axis length (exclusive).", call. = FALSE)
    }
    if (is.unsorted(sections, strictly = TRUE)) {
      stop("Split points must be strictly increasing.", call. = FALSE)
    }
    ptrs <- cpp_mlx_split(
      x$ptr,
      num_splits_ = NULL,
      indices_ = sections,
      axis = axis_idx,
      dtype_str = x$dtype,
      device_str = x$device
    )
  }

  res <- lapply(ptrs, function(ptr) .mlx_wrap_result(ptr, x$device))
  res
}

#' @noRd
.parse_pad_width <- function(pad_width, n_axes) {
  if (n_axes <= 0L) {
    stop("n_axes must be positive.", call. = FALSE)
  }

  to_matrix <- function(vals) {
    mat <- matrix(as.integer(vals), nrow = n_axes, ncol = 2, byrow = TRUE)
    mat
  }

  if (is.numeric(pad_width) && length(pad_width) == 1L) {
    return(to_matrix(rep.int(pad_width, 2L)))
  }

  if (is.numeric(pad_width) && length(pad_width) == 2L) {
    mat <- matrix(as.integer(pad_width), nrow = n_axes, ncol = 2, byrow = TRUE)
    return(mat)
  }

  if (is.matrix(pad_width)) {
    if (ncol(pad_width) != 2L) {
      stop("pad_width matrix must have two columns.", call. = FALSE)
    }
    if (nrow(pad_width) != n_axes) {
      stop("pad_width matrix must have one row per axis.", call. = FALSE)
    }
    return(matrix(as.integer(pad_width), ncol = 2))
  }

  if (is.list(pad_width)) {
    if (length(pad_width) != n_axes) {
      stop("pad_width list must have one element per axis.", call. = FALSE)
    }
    mat <- matrix(0L, nrow = n_axes, ncol = 2)
    for (i in seq_len(n_axes)) {
      vals <- pad_width[[i]]
      if (!is.numeric(vals) || length(vals) != 2L) {
        stop("Each pad_width list element must be a length-two numeric vector.", call. = FALSE)
      }
      mat[i, ] <- as.integer(vals)
    }
    return(mat)
  }

  if (is.numeric(pad_width) && length(pad_width) == 2L * n_axes) {
    return(matrix(as.integer(pad_width), nrow = n_axes, ncol = 2, byrow = TRUE))
  }

  stop("Unsupported pad_width specification.", call. = FALSE)
}

#' Reorder MLX tensor axes
#'
#' @description
#' * `mlx_moveaxis()` mirrors MLX's native [moveaxis](https://ml-explore.github.io/mlx/build/html/python/array_api/generated/mlx.core.moveaxis.html)
#'   primitive, repositioning one or more axes to new locations.
#' * `aperm.mlx()` provides the familiar R interface, permuting axes according
#'   to `perm` via repeated calls to `mlx_moveaxis()`.
#'
#' @param x,a An object coercible to `mlx` via [as_mlx()].
#' @param source Integer vector of axis indices to move (1-indexed; negatives
#'   count from the end).
#' @param destination Integer vector giving the target positions for `source`
#'   axes (1-indexed; negatives count from the end). Must be the same length as
#'   `source`.
#' @param perm Integer permutation describing the desired axis order, matching
#'   the semantics of [base::aperm()].
#' @param resize Logical flag from [base::aperm()]. Only `TRUE` is currently
#'   supported for MLX tensors.
#' @param ... Additional arguments accepted for compatibility; ignored.
#' @return An `mlx` tensor with axes permuted.
#' @export
#' @examples
#' x <- as_mlx(array(1:8, dim = c(2, 2, 2)))
#' moved <- mlx_moveaxis(x, source = 1, destination = 3)
#' permuted <- aperm(x, c(2, 1, 3))
mlx_moveaxis <- function(x, source, destination) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  if (missing(source) || missing(destination)) {
    stop("source and destination must be supplied.", call. = FALSE)
  }
  if (length(source) != length(destination)) {
    stop("source and destination must have the same length.", call. = FALSE)
  }
  if (!length(source)) {
    stop("source must contain at least one axis.", call. = FALSE)
  }

  source_idx <- .mlx_normalize_axes(source, x)
  dest_idx <- .mlx_normalize_axes(destination, x)
  source_idx <- as.integer(source_idx)
  dest_idx <- as.integer(dest_idx)
  if (anyDuplicated(source_idx)) {
    stop("source axes must be unique.", call. = FALSE)
  }
  if (anyDuplicated(dest_idx)) {
    stop("destination axes must be unique.", call. = FALSE)
  }

  ptr <- cpp_mlx_moveaxis(x$ptr, source_idx, dest_idx)
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_moveaxis
#' @export
#' @method aperm mlx
aperm.mlx <- function(a, perm = NULL, resize = TRUE, ...) {
  x <- if (is.mlx(a)) a else as_mlx(a)
  if (!isTRUE(resize)) {
    stop("`resize = FALSE` is not supported for mlx tensors.", call. = FALSE)
  }

  ndim <- length(x$dim)
  if (ndim == 0L) {
    return(x)
  }

  if (is.null(perm)) {
    perm <- rev(seq_len(ndim))
  }

  perm <- as.integer(perm)
  if (length(perm) != ndim) {
    stop("perm must have length equal to the number of dimensions.", call. = FALSE)
  }
  if (!setequal(perm, seq_len(ndim))) {
    stop("perm must be a permutation of seq_len(ndim).", call. = FALSE)
  }

  result <- x
  current <- seq_len(ndim)
  for (i in seq_len(ndim)) {
    target_axis <- perm[i]
    current_pos <- match(target_axis, current)
    if (is.na(current_pos)) {
      stop("Invalid permutation supplied.", call. = FALSE)
    }
    if (current_pos != i) {
      result <- mlx_moveaxis(result, source = current_pos, destination = i)
      axis_val <- current[current_pos]
      current <- current[-current_pos]
      current <- append(current, axis_val, after = i - 1L)
    }
  }
  result
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
