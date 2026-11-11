
#' @param value Value to assign, typically an mlx or R array. Values recycle in
#'   R's column-major order; their length must divide the selection size.
#' @rdname mlx_subset
#' @export
`[<-.mlx` <- function(x, ..., value) {
  stopifnot(is_mlx(x))
  shape <- mlx_shape(x)
  dot_expr <- as.list(substitute(alist(...)))[-1]
  idx_list <- collect_indices(dot_expr, length(shape), parent.frame())

  n_indices <- nargs() - 2L
  if (n_indices == 1L && is_matrix_index(idx_list[[1]])) {
    matrix_assign(x, idx_list[[1]], value)
  } else {
    ndim <- length(mlx_shape(x))
    if (n_indices == 1L && ndim > 1L &&
        !identical(dot_expr[[1L]], quote(expr = ))) {
      # We allow a single index, for e.g. x[x < 0] <- 0
      x_flat <- .mlx_flatten_r_order(x)
      idx <- idx_list[[1]]
      idx <- .mlx_flatten_r_order(as_mlx(idx))
      x_flat[idx] <- value
      out <- aperm(mlx_reshape(x_flat, rev(shape)))
      dimnames(out) <- dimnames(x)
      return(out)
    }
    vectors_assign(x, idx_list, value)
  }
}

is_matrix_index <- function(i1, shape) {
  i1_numeric <- if (is_mlx(i1)) mlx_dtype(i1) != "bool" else is.numeric(i1)
  # using dim() works for both matrix and mlx_matrix objects
  ! is.null(dim(i1)) &&
  i1_numeric
}

matrix_assign <- function (x, idx_mat, value) {
  if (is_mlx(idx_mat)) idx_mat <- as_r(idx_mat)
  shape <- mlx_shape(x)
  ndims <- length(shape)
  check_matrix_index(idx_mat, shape, assign = TRUE)

  if (!nrow(idx_mat)) {
    return(x)
  }

  # Convert to mlx-style zero-based indices
  idx_mat <- as_mlx(trunc(idx_mat) - 1L, dtype = "int32")

  # Per-axis indices (0-based) as mlx arrays
  coord_list <- mlx_split(idx_mat, sections = ncol(idx_mat), axis = 2L)
  coord_list <- lapply(coord_list, drop)

  check_value_fits(length(value), nrow(idx_mat))
  value <- as_mlx(value, dtype = mlx_dtype(x))
  value <- .mlx_flatten_r_order(value)
  value <- mlx_tile(value, nrow(idx_mat) %/% length(value))
  value <- mlx_reshape(value, c(nrow(idx_mat), rep(1L, ndims)))
  axes <- seq_len(ndims) - 1L

  ptr <- cpp_mlx_scatter(x$ptr, coord_list, value$ptr, axes)
  new_mlx(ptr, dimnames = dimnames(x))
}


vectors_assign <- function(x, idx_list, value) {
  shape <- mlx_shape(x)
  ndim <- length(shape)
  selection <- slice_indices(idx_list, shape)
  if (is.null(selection)) {
    idx_list <- mapply(
      function(idx, len, axis) {
        normalize_index(idx, len, assign = TRUE, names = dimnames(x)[[axis]],
                        strict_logical = ndim > 1L)
      },
      idx_list, shape, seq_along(shape), SIMPLIFY = FALSE
    )
    lens <- lengths(idx_list)
  } else {
    lens <- selection$stop - selection$start
  }
  if (any(lens == 0L)) return(x)

  target_len <- prod(lens)
  value_mlx <- as_mlx(value, dtype = mlx_dtype(x))
  val_len <- length(value_mlx)
  check_value_fits(val_len, target_len)

  if (!is.null(selection) && val_len == 1L) {
    # Native slice_update broadcasts a scalar without allocating a tiled block.
    value_mlx <- mlx_reshape(value_mlx, rep(1L, ndim))
  } else {
    value_mlx <- .mlx_flatten_r_order(value_mlx)
    value_mlx <- mlx_tile(value_mlx, target_len %/% val_len)
    # Reshape reversed axes, then transpose, to recycle in R's column-major order.
    # Scatter additionally requires one singleton slice dimension per source axis.
    target_shape <- if (is.null(selection)) c(lens, rep(1L, ndim)) else lens
    value_mlx <- aperm(mlx_reshape(value_mlx, rev(target_shape)))
  }

  if (!is.null(selection)) {
    ptr <- cpp_mlx_slice_update(
      x$ptr, value_mlx$ptr, selection$start, selection$stop, rep(1L, ndim)
    )
  } else {
    idx_norm <- lapply(idx_list, function(idx) mlx_cast(as_mlx(idx - 1L), dtype = "int32"))
    idx_grid <- mlx_meshgrid(idx_norm, sparse = TRUE, indexing = "ij")
    ptr <- cpp_mlx_scatter(x$ptr, idx_grid, value_mlx$ptr, seq_len(ndim) - 1L)
  }
  new_mlx(ptr, dimnames = dimnames(x))
}

#' Normalize an index to a standard form
#'
#' @param idx An index which may be NULL, boolean, positive or negative; and
#'  mlx or base R.
#' @param len Length of the corresponding dimension.
#' @param assign Called from subset assignment?
#' @param allow_dims Allow numeric matrix/array indices?
#' @param strict_logical Reject logical indices longer than the selected axis
#'   (required for multidimensional R indexing).
#'
#' @returns Positive index positions, or an empty vector for none.
#' @noRd
normalize_index <- function(idx, len, assign, allow_dims = FALSE, names = NULL,
                            strict_logical = FALSE) {
  if (is.null(idx)) {
    return(seq_len(len))
  }
  if (length(idx) == 0L) {
    return(integer(0))
  }
  if (anyNA(idx)) {
    stop("Index contains NA values.")
  }
  if (is.character(idx)) {
    if (is.null(names)) {
      stop("Character indices require dimnames on the selected axis.", call. = FALSE)
    }
    matched <- match(idx, names)
    if (anyNA(matched)) {
      stop("Character index not found in dimnames.", call. = FALSE)
    }
    idx <- matched
  }

  if (!is_mlx(idx) && is.numeric(idx)) idx <- trunc(idx)

  # R masks and exactly representable integer positions can be validated on the
  # host, avoiding several GPU reductions and synchronizations per axis.
  host_index <- !is_mlx(idx) && !is.object(idx) && len <= 2^24 &&
    (is.logical(idx) || (is.numeric(idx) && all(is.finite(idx)) &&
      all(abs(idx) <= 2^24)))
  if (!host_index) idx <- as_mlx(idx)
  if (is.logical(idx) || (is_mlx(idx) && identical(mlx_dtype(idx), "bool"))) {
    if (strict_logical && length(idx) > len) stop("Logical subset index out of bounds.")
    mask <- as.logical(idx)
    idx <- which(if (length(mask) < len) rep_len(mask, len) else mask)
    if (length(idx) == 0L) return(integer(0))
  }

  if (!allow_dims && !is.null(dim(idx))) {
    stop("Matrix/array subset argument. Use undimensioned vectors only.")
  }
  if (any(abs(idx) >= len + 1)) {
    stop("Subset index out of bounds.")
  }
  if (is_mlx(idx)) idx <- mlx_cast(idx, dtype = "int32")
  if (any(idx == 0)) {
    stop("Zero in subset index is not allowed.")
  }
  if (any(idx < 0)) {
    if (!all(idx < 0)) {
      stop("Mixing positive and negative subset indices is not allowed.")
    }
    idx <- setdiff(seq_len(len), abs(idx))
    if (!host_index) idx <- as_mlx(idx)
  }

  if (is_mlx(idx)) mlx_cast(idx, dtype = "int32") else as_mlx(idx, dtype = "int32")
}

check_value_fits <- function(val_len, target_len) {
  if (val_len == 0L) {
    stop("Replacement value must have length >= 1.", call. = FALSE)
  }
  if (target_len %% val_len != 0L) {
    stop("Number of items to replace is not a multiple of replacement length",
         call. = FALSE)
  }
}

# Flatten an mlx array in R's column-major order
.mlx_flatten_r_order <- function(x) {
  ptr <- cpp_mlx_flatten_r_order(x$ptr)
  out <- new_mlx(ptr)
  mlx_reshape(out, length(x))
}
