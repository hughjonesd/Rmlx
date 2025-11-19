# Internal casting helper ----------------------------------------------------

#' Cast an mlx array to a different dtype or device
#'
#' @param x mlx array to cast.
#' @param dtype Target dtype string. Defaults to the array's current dtype.
#' @param device Target device string. Defaults to the array's current device.
#' @return An mlx array with the requested dtype and device.
#' @noRd
.mlx_cast <- function(x, dtype = NULL, device = x$device) {
  if (!inherits(x, "mlx")) {
    stop("Expected an mlx array.", call. = FALSE)
  }
  current_dtype <- mlx_dtype(x)
  if (is.null(dtype)) {
    dtype <- current_dtype
  }
  if (identical(dtype, current_dtype) && identical(device, x$device)) {
    return(x)
  }
  ptr <- cpp_mlx_cast(x$ptr, dtype, device)
  new_mlx(ptr, device)
}

#' Normalize axes for insertion operations
#'
#' @param axes Integer vector of 1-indexed target axes for the new dimensions.
#' @param dims Integer vector of the current dimensions.
#' @return Integer vector of 0-indexed axes suitable for MLX.
#' @noRd
.mlx_normalize_new_axes <- function(axes, dims) {
  if (length(axes) == 0L) {
    stop("axes must contain at least one element.", call. = FALSE)
  }
  axes <- as.integer(axes)
  if (any(is.na(axes))) {
    stop("axes cannot contain NA values.", call. = FALSE)
  }
  result_ndim <- length(dims) + 1L
  if (any(axes < 1L | axes > result_ndim)) {
    stop("axes must be between 1 and length(dim) + 1.", call. = FALSE)
  }
  axes0 <- axes - 1L
  if (any(axes0 >= result_ndim)) {
    stop("axis values are out of bounds.", call. = FALSE)
  }
  sort(unique(axes0))
}

#' Stack mlx arrays along a new axis
#'
#' @param ... One or more arrays (or a single list of arrays) coercible to mlx.
#' @param axis Position of the new axis (1-indexed). Supply values between 1 and
#'   `length(dim(x)) + 1` to insert anywhere along the dimension list.
#' @return An mlx array with one additional dimension.
#' @seealso [mlx.core.stack](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.stack)
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' y <- as_mlx(matrix(5:8, 2, 2))
#' stacked <- mlx_stack(x, y, axis = 1)
mlx_stack <- function(..., axis = 1L) {
  arrays <- list(...)
  if (length(arrays) == 1L && is.list(arrays[[1]]) && !is.mlx(arrays[[1]])) {
    arrays <- arrays[[1]]
  }
  if (!length(arrays)) {
    stop("No arrays supplied.", call. = FALSE)
  }
  arrays <- lapply(arrays, as_mlx)
  dtypes <- lapply(arrays, mlx_dtype)
  dtype <- Reduce(.promote_dtype, dtypes)
  device <- Reduce(.common_device, lapply(arrays, `[[`, "device"))
  arrays <- lapply(arrays, .mlx_cast, dtype = dtype, device = device)

  axis_vec <- .mlx_normalize_new_axes(axis, dim(arrays[[1]]))
  if (length(axis_vec) != 1L) {
    stop("`axis` must be a single insertion position.", call. = FALSE)
  }
  axis0 <- axis_vec
  ptr <- cpp_mlx_stack(arrays, axis0, device)
  new_mlx(ptr, device)
}

#' Drop singleton dimensions
#'
#' `drop()` removes axes of length one. For base R objects this dispatches to
#' [base::drop()], while `drop.mlx()` delegates to [mlx_squeeze()] so that mlx
#' arrays remain on the device.
#'
#' @param x Object to drop dimensions from.
#' @return An object with singleton dimensions removed. For mlx inputs the
#'   result is another mlx array.
#' @seealso [mlx_squeeze()], [base::drop()]
#' @export
drop <- function(x) {
  UseMethod("drop")
}

#' @rdname drop
#' @export
drop.default <- function(x) base::drop(x)

#' @rdname drop
#' @export
drop.mlx <- function(x) {
  mlx_squeeze(x)
}

#' Remove singleton dimensions
#'
#' @inheritParams mlx_array_required
#' @param axes Optional integer vector of axes (1-indexed) to remove. When `NULL`
#'   all axes of length one are removed.
#' @return An mlx array with the selected axes removed.
#' @seealso [mlx.core.squeeze](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.squeeze)
#' @export
#' @examples
#' x <- as_mlx(array(1:4, dim = c(1, 2, 2, 1)))
#' mlx_squeeze(x)
#' mlx_squeeze(x, axes = 1)
mlx_squeeze <- function(x, axes = NULL) {
  x <- as_mlx(x)
  if (is.null(axes)) {
    ptr <- cpp_mlx_squeeze(x$ptr, NULL)
  } else {
    axes_idx <- .mlx_normalize_axes(axes, x)
    ptr <- cpp_mlx_squeeze(x$ptr, axes_idx)
  }
  new_mlx(ptr, x$device)
}

#' Insert singleton dimensions
#'
#' @inheritParams mlx_array_required
#' @param axes Integer vector of axis positions (1-indexed) where new singleton
#'   dimensions should be inserted.
#' @return An mlx array with additional dimensions of length one.
#' @seealso [mlx.core.expand_dims](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.expand_dims)
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_expand_dims(x, axes = 1)
mlx_expand_dims <- function(x, axes) {
  x <- as_mlx(x)
  axes0 <- .mlx_normalize_new_axes(axes, dim(x))
  ptr <- cpp_mlx_expand_dims(x$ptr, axes0)
  new_mlx(ptr, x$device)
}

#' Repeat array elements
#'
#' @inheritParams mlx_array_required
#' @param repeats Number of repetitions.
#' @param axis Optional axis along which to repeat. When `NULL`, the array is
#'   flattened before repetition (matching NumPy semantics).
#' @return An mlx array with repeated values.
#' @seealso [mlx.core.repeat](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.repeat)
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_repeat(x, repeats = 2, axis = 2)
mlx_repeat <- function(x, repeats, axis = NULL) {
  x <- as_mlx(x)
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
  new_mlx(ptr, x$device)
}

#' Tile an array
#'
#' @inheritParams mlx_array_required
#' @param reps Integer vector giving the number of repetitions for each axis.
#' @return An mlx array with tiled content.
#' @seealso [mlx.core.tile](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.tile)
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_tile(x, reps = c(1, 2))
mlx_tile <- function(x, reps) {
  x <- as_mlx(x)
  reps <- as.integer(reps)
  if (length(reps) == 0L || any(is.na(reps)) || any(reps <= 0L)) {
    stop("reps must be positive integers.", call. = FALSE)
  }
  ptr <- cpp_mlx_tile(x$ptr, reps)
  new_mlx(ptr, x$device)
}

#' Roll array elements
#'
#' @inheritParams mlx_array_required
#' @param shift Integer vector giving the number of places by which elements are shifted.
#' @param axes Optional integer vector (1-indexed) along which elements are shifted.
#'   When `NULL`, the array is flattened and shifted.
#' @return An mlx array with elements circularly shifted.
#' @seealso [mlx.core.roll](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.roll)
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_roll(x, shift = 1, axes = 2)
mlx_roll <- function(x, shift, axes = NULL) {
  x <- as_mlx(x)
  shift <- as.integer(shift)
  if (length(shift) == 0L || any(is.na(shift))) {
    stop("shift must contain integer values.", call. = FALSE)
  }

  if (is.null(axes)) {
    ptr <- cpp_mlx_roll(x$ptr, shift, NULL)
  } else {
    axes0 <- .mlx_normalize_axes(axes, x)
    if (length(shift) != length(axes0)) {
      stop("shift and axes must have the same length.", call. = FALSE)
    }
    ptr <- cpp_mlx_roll(x$ptr, shift, axes0)
  }
  new_mlx(ptr, x$device)
}

#' Pad or split mlx arrays
#'
#' @description
#' * `mlx_pad()` mirrors the MLX padding primitive, enlarging each axis according
#'   to `pad_width`. Values are added symmetrically (`pad_width[i, 1]` before,
#'   `pad_width[i, 2]` after) using the specified `mode`.
#' * `mlx_split()` divides an array along an axis either into equal sections
#'   (`sections` scalar) or at explicit 1-based split points (`sections` vector),
#'   returning a list of mlx arrays.
#'
#' @inheritParams common_params
#' @param pad_width Padding extents. Supply a single integer, a length-two
#'   numeric vector, or a matrix/list with one `(before, after)` pair per padded
#'   axis.
#' @param value Constant fill value used when `mode = "constant"`.
#' @param mode Padding mode passed to MLX (e.g., `"constant"`, `"edge"`,
#'   `"reflect"`).
#' @param axes Optional integer vector of axes (1-indexed) to which `pad_width`
#'   applies. Unlisted axes receive zero padding.
#' @param sections Either a single integer (number of equal parts) or an integer
#'   vector of 1-based split points along `axis`.
#' @param axis Axis (1-indexed) to operate on.
#' @return For `mlx_pad()`, an mlx array; for `mlx_split()`, a list of mlx
#'   arrays.
#' @seealso [mlx.core.pad](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.pad)
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
  x <- as_mlx(x)
  x_dtype <- mlx_dtype(x)
  mode <- match.arg(mode)

  ndim <- length(dim(x))
  if (ndim == 0L) {
    stop("Cannot pad a scalar mlx array.", call. = FALSE)
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
    x_dtype,
    x$device,
    mode
  )
  new_mlx(ptr, x$device)
}

#' @rdname mlx_pad
#' @seealso [mlx.core.split](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.split)
#' @export
mlx_split <- function(x, sections, axis = 1L) {
  x <- as_mlx(x)
  x_dtype <- mlx_dtype(x)
  if (missing(sections)) {
    stop("sections must be supplied.", call. = FALSE)
  }
  axis_idx <- .mlx_normalize_axis(axis, x)
  dim_len <- dim(x)[axis_idx + 1L]

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
      dtype_str = x_dtype,
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
      dtype_str = x_dtype,
      device_str = x$device
    )
  }

  res <- lapply(ptrs, function(ptr) new_mlx(ptr, x$device))
  res
}

#' Split mlx arrays along a margin
#'
#' `asplit()` extends base [asplit()] to work with mlx arrays by delegating to
#' [mlx_split()]. When `x` is mlx the result is a list of mlx arrays; otherwise,
#' the base implementation is used.
#'
#' Currently only a single `MARGIN` value is supported for mlx arrays.
#'
#' @inheritParams base::asplit
#' @returns For mlx inputs, a list of mlx arrays; otherwise matches
#'   [base::asplit()].
#' @export
asplit <- function(x, MARGIN, drop = FALSE) {
  UseMethod("asplit")
}

#' @rdname asplit
#' @export
asplit.default <- function(x, MARGIN, drop = FALSE) {
  base::asplit(x, MARGIN, drop = drop)
}

#' @rdname asplit
#' @export
asplit.mlx <- function(x, MARGIN, drop = FALSE) {
  x <- as_mlx(x)
  if (length(MARGIN) != 1L) {
    stop("asplit() for mlx arrays currently supports a single margin.", call. = FALSE)
  }
  axis_idx <- .mlx_normalize_axis(MARGIN, x)
  dim_len <- dim(x)[axis_idx + 1L]
  splits <- mlx_split(x, sections = dim_len, axis = MARGIN)

  rest_dim <- dim(x)[-(axis_idx + 1L)]
  splits <- lapply(splits, function(part) {
    if (isTRUE(drop)) {
      dim(part) <- integer(0L)
    } else {
      dim(part) <- if (length(rest_dim)) rest_dim else integer(0L)
    }
    part
  })
  splits
}

#' Parse padding specification into matrix format
#'
#' @param pad_width Scalar, length-2 vector, matrix, list, or full-length vector.
#' @param n_axes Integer number of axes being padded.
#' @return Integer matrix with n_axes rows and 2 columns (before, after).
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
    if (any(!vapply(pad_width, is.numeric, logical(1)))) {
      stop("Each pad_width list element must be a length-two numeric vector.", call. = FALSE)
    }
    if (any(lengths(pad_width) != 2L)) {
      stop("Each pad_width list element must be a length-two numeric vector.", call. = FALSE)
    }
    mat <- matrix(as.integer(unlist(pad_width)), nrow = n_axes, ncol = 2, byrow = TRUE)
    return(mat)
  }

  if (is.numeric(pad_width) && length(pad_width) == 2L * n_axes) {
    return(matrix(as.integer(pad_width), nrow = n_axes, ncol = 2, byrow = TRUE))
  }

  stop("Unsupported pad_width specification.", call. = FALSE)
}

#' Reorder mlx array axes
#'
#' @description
#' * `mlx_moveaxis()` repositions one or more axes to new locations.
#' * `aperm.mlx()` provides the familiar R interface, permuting axes according
#'   to `perm` via repeated calls to `mlx_moveaxis()`.
#'
#' @param x,a An object coercible to mlx via [as_mlx()].
#' @param source Integer vector of axis indices to move (1-indexed).
#' @param destination Integer vector giving the target positions for `source`
#'   axes (1-indexed). Must be the same length as `source`.
#' @param perm Integer permutation describing the desired axis order, matching
#'   the semantics of [base::aperm()].
#' @param resize Logical flag from [base::aperm()]. Only `TRUE` is currently
#'   supported for mlx arrays.
#' @param ... Additional arguments accepted for compatibility; ignored.
#' @return An mlx array with axes permuted.
#' @seealso [mlx.core.moveaxis](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.moveaxis)
#' @export
#' @examples
#' x <- as_mlx(array(1:8, dim = c(2, 2, 2)))
#' moved <- mlx_moveaxis(x, source = 1, destination = 3)
#' permuted <- aperm(x, c(2, 1, 3))
mlx_moveaxis <- function(x, source, destination) {
  x <- as_mlx(x)
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
  new_mlx(ptr, x$device)
}

#' @rdname mlx_moveaxis
#' @export
aperm.mlx <- function(a, perm = NULL, resize = TRUE, ...) {
  x <- as_mlx(a)
  if (!isTRUE(resize)) {
    stop("`resize = FALSE` is not supported for mlx arrays.", call. = FALSE)
  }

  ndim <- length(dim(x))
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

#' Ensure contiguous memory layout
#'
#' Returns a copy of `x` with contiguous strides on the requested device or stream.
#'
#' @inheritParams mlx_array_required
#' @inheritParams common_params
#' @return An mlx array backed by contiguous storage on the specified device.
#' @seealso <https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.contiguous>
#' @export
#' @examples
#' x <- mlx_swapaxes(as_mlx(matrix(1:4, 2, 2)), axis1 = 1, axis2 = 2)
#' y <- mlx_contiguous(x)
#' identical(as.array(x), as.array(y))
mlx_contiguous <- function(x, device = NULL) {
  x <- as_mlx(x)
  target <- if (is.null(device)) x$device else device
  handle <- .mlx_resolve_device(target, x$device)
  ptr <- .mlx_eval_with_stream(handle, function(dev) cpp_mlx_contiguous(x$ptr, dev))
  new_mlx(ptr, handle$device)
}

#' Flatten axes of an mlx array
#'
#' `mlx_flatten()` mirrors [`mlx.core.flatten()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.flatten),
#' collapsing a contiguous range of axes into a single dimension.
#'
#' @inheritParams mlx_array_required
#' @param start_axis First axis (1-indexed) in the flattened range.
#' @param end_axis Last axis (1-indexed) in the flattened range. Omit to use the
#'   final dimension.
#' @return An mlx array with the selected axes collapsed.
#' @seealso [mlx.core.flatten](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.flatten)
#' @export
#' @examples
#' x <- as_mlx(array(1:12, dim = c(2, 3, 2)))
#' mlx_flatten(x)
#' mlx_flatten(x, start_axis = 2, end_axis = 3)
mlx_flatten <- function(x, start_axis = 1L, end_axis = NULL) {
  x <- as_mlx(x)

  if (length(dim(x)) == 0L) {
    return(x)
  }

  start_axis <- as.integer(start_axis)
  if (length(start_axis) != 1L || is.na(start_axis)) {
    stop("start_axis must be a single positive integer.", call. = FALSE)
  }
  if (missing(end_axis) || is.null(end_axis)) {
    end_axis <- length(dim(x))
  }
  end_axis <- as.integer(end_axis)
  if (length(end_axis) != 1L || is.na(end_axis)) {
    stop("end_axis must be NULL or a single positive integer.", call. = FALSE)
  }

  start_idx <- .mlx_normalize_axis_single(start_axis, x)
  end_idx <- .mlx_normalize_axis_single(end_axis, x)

  if (start_idx > end_idx) {
    stop("start_axis must be less than or equal to end_axis.", call. = FALSE)
  }

  ptr <- cpp_mlx_flatten(x$ptr, start_idx, end_idx)
  new_mlx(ptr, x$device)
}

#' Swap two axes of an mlx array
#'
#' `mlx_swapaxes()` mirrors [`mlx.core.swapaxes()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.swapaxes),
#' exchanging two dimensions while leaving others intact.
#'
#' @inheritParams mlx_array_required
#' @param axis1,axis2 Axes to swap (1-indexed).
#' @return An mlx array with the specified axes exchanged.
#' @seealso [mlx.core.swapaxes](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.swapaxes)
#' @export
#' @examples
#' x <- as_mlx(array(1:24, dim = c(2, 3, 4)))
#' swapped <- mlx_swapaxes(x, axis1 = 1, axis2 = 3)
#' dim(swapped)
mlx_swapaxes <- function(x, axis1, axis2) {
  x <- as_mlx(x)

  if (missing(axis1) || missing(axis2)) {
    stop("axis1 and axis2 must be supplied.", call. = FALSE)
  }

  axis1 <- as.integer(axis1)
  axis2 <- as.integer(axis2)

  axis1_idx <- .mlx_normalize_axis_single(axis1, x)
  axis2_idx <- .mlx_normalize_axis_single(axis2, x)

  ptr <- cpp_mlx_swapaxes(x$ptr, axis1_idx, axis2_idx)
  new_mlx(ptr, x$device)
}

#' Construct coordinate arrays from input vectors
#'
#' `mlx_meshgrid()` mirrors [`mlx.core.meshgrid()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.meshgrid),
#' returning coordinate arrays suitable for vectorised evaluation on MLX devices.
#'
#' @param ... One or more arrays (or a single list) convertible via [as_mlx()] representing coordinate vectors.
#' @param sparse Logical flag producing broadcast-friendly outputs when `TRUE`.
#' @param indexing Either `"xy"` (Cartesian) or `"ij"` (matrix) indexing.
#' @inheritParams common_params
#' @return A list of mlx arrays matching the number of inputs.
#' @seealso [mlx.core.meshgrid](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.meshgrid)
#' @export
#' @examples
#' xs <- as_mlx(1:3)
#' ys <- as_mlx(1:2)
#' mlx_meshgrid(xs, ys, indexing = "xy")
mlx_meshgrid <- function(...,
                         sparse = FALSE,
                         indexing = c("xy", "ij"),
                         device = NULL) {
  arrays <- list(...)
  if (length(arrays) == 1L && is.list(arrays[[1]]) && !is.mlx(arrays[[1]])) {
    arrays <- arrays[[1]]
  }
  if (!length(arrays)) {
    stop("No arrays supplied.", call. = FALSE)
  }
  arrays <- lapply(arrays, as_mlx)

  dtypes <- lapply(arrays, mlx_dtype)
  dtype <- Reduce(.promote_dtype, dtypes)
  default_device <- Reduce(.common_device, lapply(arrays, `[[`, "device"))
  target <- if (is.null(device)) default_device else device
  handle <- .mlx_resolve_device(target, default_device)
  arrays <- lapply(arrays, .mlx_cast, dtype = dtype, device = handle$device)

  indexing <- match.arg(indexing)
  ptrs <- .mlx_eval_with_stream(handle, function(dev) cpp_mlx_meshgrid(arrays, sparse, indexing, dev))
  lapply(ptrs, function(ptr) new_mlx(ptr, handle$device))
}

#' Broadcast an array to a new shape
#'
#' `mlx_broadcast_to()` mirrors [`mlx.core.broadcast_to()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.broadcast_to),
#' repeating singleton dimensions without copying data.
#'
#' @inheritParams mlx_array_required
#' @param shape Integer vector describing the broadcasted shape.
#' @inheritParams common_params
#' @return An mlx array with the requested dimensions.
#' @seealso [mlx.core.broadcast_to](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.broadcast_to)
#' @export
#' @examples
#' x <- mlx_matrix(1:3, nrow = 1)
#' broadcast <- mlx_broadcast_to(x, c(5, 3))
#' dim(broadcast)
mlx_broadcast_to <- function(x, shape, device = NULL) {
  x <- as_mlx(x)
  shape <- .validate_shape(shape)
  target <- if (is.null(device)) x$device else device
  handle <- .mlx_resolve_device(target, x$device)

  ptr <- .mlx_eval_with_stream(handle, function(dev) cpp_mlx_broadcast_to(x$ptr, shape, dev))
  new_mlx(ptr, handle$device)
}

#' Broadcast multiple arrays to a shared shape
#'
#' `mlx_broadcast_arrays()` mirrors [`mlx.core.broadcast_arrays()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.broadcast_arrays),
#' returning a list of inputs expanded to a common shape.
#'
#' @param ... One or more arrays (or a single list) convertible via [as_mlx()].
#' @inheritParams common_params
#' @return A list of broadcast mlx arrays.
#' @seealso [mlx.core.broadcast_arrays](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.broadcast_arrays)
#' @export
#' @examples
#' a <- as_mlx(matrix(1:3, nrow = 1))
#' b <- as_mlx(matrix(1:3, ncol = 1))
#' outs <- mlx_broadcast_arrays(a, b)
#' lapply(outs, dim)
mlx_broadcast_arrays <- function(..., device = NULL) {
  arrays <- list(...)
  if (length(arrays) == 1L && is.list(arrays[[1]]) && !is.mlx(arrays[[1]])) {
    arrays <- arrays[[1]]
  }
  if (!length(arrays)) {
    stop("No arrays supplied.", call. = FALSE)
  }
  arrays <- lapply(arrays, as_mlx)

  dtypes <- lapply(arrays, mlx_dtype)
  dtype <- Reduce(.promote_dtype, dtypes)
  default_device <- Reduce(.common_device, lapply(arrays, `[[`, "device"))
  target <- if (is.null(device)) default_device else device
  handle <- .mlx_resolve_device(target, default_device)
  arrays <- lapply(arrays, .mlx_cast, dtype = dtype, device = handle$device)

  ptrs <- .mlx_eval_with_stream(handle, function(dev) cpp_mlx_broadcast_arrays(arrays, dev))
  lapply(ptrs, function(ptr) new_mlx(ptr, handle$device))
}

#' Elementwise conditional selection
#'
#' @param condition Logical mlx array (non-zero values are treated as `TRUE`).
#' @param x,y Arrays broadcastable to the shape of `condition`.
#' @return An mlx array where elements are drawn from `x` when
#'   `condition` is `TRUE`, otherwise from `y`.
#' @details Behaves like [ifelse()] for arrays, but evaluates both branches.
#' @seealso [mlx.core.where](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.where)
#' @export
#' @examples
#' cond <- mlx_matrix(c(TRUE, FALSE, TRUE, FALSE), 2, 2)
#' a <- mlx_matrix(1:4, 2, 2)
#' b <- mlx_matrix(5:8, 2, 2)
#' mlx_where(cond, a, b)
mlx_where <- function(condition, x, y) {
  condition <- as_mlx(condition)
  x <- as_mlx(x)
  y <- as_mlx(y)

  x_dtype <- mlx_dtype(x)
  y_dtype <- mlx_dtype(y)
  result_dtype <- .promote_dtype(x_dtype, y_dtype)
  result_device <- .common_device(x$device, y$device)

  condition <- .mlx_cast(condition, dtype = "bool", device = result_device)
  x <- .mlx_cast(x, dtype = result_dtype, device = result_device)
  y <- .mlx_cast(y, dtype = result_dtype, device = result_device)

  ptr <- cpp_mlx_where(condition$ptr, x$ptr, y$ptr, result_dtype, result_device)
  new_mlx(ptr, result_device)
}
