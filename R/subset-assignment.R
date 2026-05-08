
#' @param value Value to assign, typically an mlx or R array
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
    if (n_indices == 1L && ndim > 1L) {
      # We allow a single index, for e.g. x[x < 0] <- 0
      x_flat <- mlx_flatten(x)
      idx <- idx_list[[1]]
      idx <- mlx_flatten(as_mlx(idx))
      x_flat[idx] <- value
      return(mlx_reshape(x_flat, shape))
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
  idx_mat <- as_r(idx_mat)
  shape <- mlx_shape(x)
  ndims <- length(shape)
  check_matrix_index(idx_mat, shape, assign = TRUE)

  if (!nrow(idx_mat)) {
    return(x)
  }

  # Convert to mlx-style zero-based indices
  idx_mat <- as_mlx(idx_mat - 1L, dtype = "int32", device = mlx_device(x))

  # Per-axis indices (0-based) as mlx arrays
  coord_list <- mlx_split(idx_mat, sections = ncol(idx_mat), axis = 2L)
  coord_list <- lapply(coord_list, drop)

  check_value_fits(length(value), nrow(idx_mat))
  # if (duplicated_rows_lex(idx_mat)) {
  #   stop("Duplicate indices are not allowed in assignment.", call. = FALSE)
  # }
  value <- as_mlx(value, dtype = mlx_dtype(x), device = mlx_device(x))
  value <- mlx_repeat(value, nrow(idx_mat) %/% length(value))
  value <- mlx_reshape(value, c(nrow(idx_mat), rep(1L, ndims)))
  axes <- seq_len(ndims) - 1L

  ptr <- cpp_mlx_scatter(x$ptr, coord_list, value$ptr, axes, mlx_device(x))
  new_mlx(ptr, mlx_device(x))
}


vectors_assign <- function(x, idx_list, value) {
  shape <- mlx_shape(x)
  ndim <- length(shape)
  idx_list <- mapply(normalize_index, idx_list, shape, SIMPLIFY = FALSE,
                     MoreArgs = list(assign = TRUE))

  if (any(vapply(idx_list, is.null, integer(1)))) {
    return(x)
  }
  idx_norm <- lapply(idx_list, function(idx) idx - 1L)
  lens <- lengths(idx_norm)

  target_len <- prod(lens)
  value_mlx <- as_mlx(value, dtype = mlx_dtype(x), device = mlx_device(x))
  val_len <- length(value_mlx)
  check_value_fits(val_len, target_len)

  tiles <- target_len %/% val_len
  value_mlx <- .mlx_flatten_r_order(value_mlx)
  value_mlx <- mlx_tile(value_mlx, tiles)
  # deep magic, so think hard. Idea is to get back to R order (but also prepare
  # for scatter below)
  rev_shape <- rev(c(lens, rep(1L, ndim)))
  value_mlx <- mlx_reshape(value_mlx, rev_shape)
  value_mlx <- aperm(value_mlx)

  idx_grid <- mlx_meshgrid(idx_norm, sparse = FALSE, indexing = "ij", device = x$device)
  idx_grid <- lapply(idx_grid, mlx_cast, dtype = "int32")
  axes <- seq_len(ndim) - 1L
  ptr <- cpp_mlx_scatter(x$ptr, idx_grid, value_mlx$ptr, axes, x$device)
  new_mlx(ptr, mlx_device(x))
}

#' Normalize an index to a standard form
#'
#' @param idx An index which may be NULL, boolean, positive or negative; and
#'  mlx or base R.
#' @param len Length of the corresponding dimension.
#' @param assign Called from subset assignment? If so duplicates throw an error.
#' @param allow_dims Allow matrix/array indices?
#'
#' @returns A positive mlx vector of index positions, or NULL for none.
#' @noRd
normalize_index <- function(idx, len, assign, allow_dims = FALSE) {
  if (is.null(idx)) {
    return(seq_len(len))
  }
  if (length(idx) == 0L) {
    return(NULL)
  }
  if (anyNA(idx)) {
    stop("Index contains NA values.")
  }

  idx <- as_mlx(idx)
  if (identical(mlx_dtype(idx), "bool")) {
    idx <- which(as.logical(idx))
    if (length(idx) == 0L) return(NULL)
    idx <- mlx_vector(idx)
  }
  if (! allow_dims && ! is.null(dim(idx))) {
    stop("Matrix/array subset argument. Use undimensioned vectors only.")
  }

  if (any(idx == 0)) {
    stop("Zero in subset index is not allowed.")
  }
  if (any(abs(idx) > len)) {
    stop("Subset index out of bounds.")
  }
  if (any(idx < 0)) {
    if (!all(idx < 0)) {
      stop("Mixing positive and negative subset indices is not allowed.")
    }
    idx <- setdiff(seq_len(len), abs(idx))
    idx <- as_mlx(idx)
  }

  # if (assign && anyDuplicated(idx) > 0L) {
  #   stop("Duplicate indices in subset assignment.")
  # }

  idx <- mlx_cast(idx, dtype = "int32")
  idx
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
  ptr <- cpp_mlx_flatten_r_order(x$ptr, x$device)
  out <- new_mlx(ptr, x$device)
  mlx_reshape(out, length(x))
}
