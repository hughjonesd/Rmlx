
#' Wrap a raw MLX pointer into an mlx object
#'
#' @param ptr External pointer returned by C++ bindings.
#' @param device Device string associated with the array.
#' @return An mlx array.
#' @noRd
.mlx_wrap_result <- function(ptr, device) {
  dim <- cpp_mlx_shape(ptr)
  dtype <- cpp_mlx_dtype(ptr)
  new_mlx(ptr, dim, dtype, device)
}

.mlx_is_stream <- function(x) inherits(x, "mlx_stream")

.mlx_resolve_device <- function(device, default = mlx_default_device()) {
  if (missing(device) || is.null(device)) {
    device <- default
  }

  if (.mlx_is_stream(device)) {
    return(list(device = device$device, stream_ptr = device$ptr))
  }

  if (!is.character(device) || length(device) != 1L) {
    stop('device must be "gpu", "cpu", or an mlx_stream.', call. = FALSE)
  }

  device_chr <- match.arg(device, c("gpu", "cpu"))
  list(device = device_chr, stream_ptr = NULL)
}

.mlx_eval_with_stream <- function(handle, fn) {
  if (is.null(handle$stream_ptr)) {
    return(fn(handle$device))
  }

  old <- cpp_mlx_stream_default(handle$device)
  on.exit(cpp_mlx_set_default_stream(old), add = TRUE)
  cpp_mlx_set_default_stream(handle$stream_ptr)
  fn(handle$device)
}

#' Common parameters for MLX array creation
#'
#' @param dim Integer vector specifying the array shape/dimensions.
#' @param dtype Character string specifying the MLX data type. Common options:
#'   - Floating point: `"float32"`, `"float64"`
#'   - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`, `"uint16"`,
#'     `"uint32"`, `"uint64"`
#'   - Other: `"bool"`, `"complex64"`
#'
#'   Supported types vary by function; see individual function documentation.
#' @param device Execution target: provide `"gpu"`, `"cpu"`, or an
#'   `mlx_stream` created via [mlx_new_stream()]. Defaults to the current
#'   [mlx_default_device()].
#' @name mlx_creation_params
#' @keywords internal
NULL

#' Print MLX array
#'
#' @inheritParams common_params
#' @param ... Additional arguments (ignored)
#' @export
#' @method print mlx
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' print(x)
print.mlx <- function(x, ...) {
  cat(sprintf("mlx array [%s]\n", paste(x$dim, collapse = " x ")))
  cat(sprintf("  dtype: %s\n", x$dtype))
  cat(sprintf("  device: %s\n", x$device))

  # Show preview for small arrays
  total_size <- prod(x$dim)
  if (total_size <= 100 && length(x$dim) <= 2) {
    cat("  values:\n")
    mat <- as.matrix(x)
    print(mat)
  } else {
    cat(sprintf("  (%d elements, not shown)\n", total_size))
  }

  invisible(x)
}

#' Object structure for MLX array
#'
#' @param object An mlx object
#' @param ... Additional arguments (ignored)
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' str(x)
str.mlx <- function(object, ...) {
  cat(sprintf(
    "mlx [%s] %s on %s\n",
    paste(object$dim, collapse = " x "),
    object$dtype,
    object$device
  ))
  invisible(NULL)
}

#' Get dimensions of MLX array
#'
#' @inheritParams common_params
#' @return Integer vector of dimensions
#' @export
#' @method dim mlx
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' dim(x)
dim.mlx <- function(x) {
  x$dim
}

#' Set dimensions of MLX array
#'
#' Reshapes the MLX array to the specified dimensions. The total number of
#' elements must remain the same.
#'
#' @inheritParams common_params
#' @param value Integer vector of new dimensions
#' @return Reshaped mlx object
#' @export
#' @method dim<- mlx
#' @examples
#' x <- as_mlx(1:12)
#' dim(x) <- c(3, 4)
#' dim(x)
`dim<-.mlx` <- function(x, value) {
  if (!is.numeric(value) || any(is.na(value))) {
    stop("dims must be a numeric vector without NAs", call. = FALSE)
  }

  value <- as.integer(value)

  if (any(value < 0)) {
    stop("dims cannot be negative", call. = FALSE)
  }

  # Check that product matches
  current_size <- prod(x$dim)
  new_size <- prod(value)

  if (current_size != new_size) {
    stop(sprintf(
      "dims [product %d] do not match the length of object [%d]",
      new_size, current_size
    ), call. = FALSE)
  }

  mlx_reshape(x, value)
}

#' Reshape an mlx array
#'
#' @inheritParams mlx_array_required
#' @param newshape Integer vector specifying the new dimensions.
#' @return An mlx array with the specified shape.
#' @seealso [mlx.core.reshape](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.reshape.html)
#' @export
#' @examples
#' x <- as_mlx(1:12)
#' mlx_reshape(x, c(3, 4))
#' mlx_reshape(x, c(2, 6))
mlx_reshape <- function(x, newshape) {
  x <- as_mlx(x)

  if (!is.numeric(newshape) || any(is.na(newshape))) {
    stop("newshape must be a numeric vector without NAs", call. = FALSE)
  }

  newshape <- as.integer(newshape)

  if (any(newshape < 0)) {
    stop("newshape cannot contain negative values", call. = FALSE)
  }

  current_size <- prod(x$dim)
  new_size <- prod(newshape)

  if (current_size != new_size) {
    stop(sprintf(
      "Cannot reshape array of size %d into shape with size %d",
      current_size, new_size
    ), call. = FALSE)
  }

  ptr <- cpp_mlx_reshape(x$ptr, newshape)
  dim_result <- cpp_mlx_shape(ptr)
  dtype_result <- cpp_mlx_dtype(ptr)
  new_mlx(ptr, dim_result, dtype_result, x$device)
}

#' Get length of MLX array
#'
#' @inheritParams common_params
#' @return Total number of elements
#' @export
#' @method length mlx
#' @examples
#' x <- as_mlx(matrix(1:6, 2, 3))
#' length(x)
length.mlx <- function(x) {
  prod(x$dim)
}

#' Get dimensions helper
#'
#' @inheritParams common_params
#' @return Dimensions
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 2, 3))
#' mlx_dim(x)
mlx_dim <- function(x) {
  stopifnot(is.mlx(x))
  x$dim
}

#' Get data type helper
#'
#' @inheritParams common_params
#' @return Data type string
#' @export
#' @examples
#' x <- as_mlx(matrix(1:6, 2, 3))
#' mlx_dtype(x)
mlx_dtype <- function(x) {
  stopifnot(is.mlx(x))
  x$dtype
}

#' Subset MLX array
#'
#' MLX subsetting mirrors base R for the common cases while avoiding a few of
#' the language's historical footguns:
#'
#' * **Numeric indices**: positive (1-based) and purely negative vectors are
#'   supported. Negative indices drop the listed elements, just as in base R.
#'   Mixing signs is an error and `0` is not allowed.
#' * **Logical indices**: recycled to the target dimension length. Logical masks
#'   may be mixed with numeric indices across dimensions.
#' * **Matrices/arrays**: numeric matrices (or higher dimensional arrays) select
#'   individual elements, one coordinate per row. The trailing dimension must
#'   match the array rank and entries must be positive; negative matrices are
#'   rejected to avoid ambiguous complements.
#' * **`mlx` indices**: `mlx` vectors, logical masks, and matrices behave the
#'   same as their R equivalents. One-dimensional MLX arrays are treated as
#'   vectors rather than 1-column matrices.
#' * **`drop`**: dimensions are preserved by default (`drop = FALSE`), matching
#'   the package's preference for explicit shapes.
#' * **Unsupported**: character indices and named lookups are not implemented.
#'
#' @inheritParams common_params
#' @param ... Indices for each dimension. Provide one per axis; omitted indices
#'   select the full extent. Logical indices recycle to the dimension length.
#' @param drop Should dimensions be dropped? (default: FALSE)
#' @param value Replacement value(s) for `\code{[<-}` (scalar, vector, matrix,
#'   or array) recycled to match the selection.
#' @return Subsetted mlx object
#' @seealso [mlx.core.take](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.take)
#' @name mlx_subset
#' @importFrom utils tail
#' @export
#' @method [ mlx
#' @examples
#' x <- as_mlx(matrix(1:9, 3, 3))
#' x[1, ]
`[.mlx` <- function(x, ..., drop = FALSE) {
  ndim <- length(x$dim)
  if (ndim == 0L) {
    stop("Cannot subset a scalar mlx array.", call. = FALSE)
  }

  idx_list <- vector("list", ndim)
  dot_expr <- as.list(substitute(alist(...)))[-1]

  if (length(dot_expr) > ndim) {
    stop("Incorrect number of indices supplied.", call. = FALSE)
  }

  if (length(dot_expr)) {
    for (k in seq_along(dot_expr)) {
      expr <- dot_expr[[k]]
      value <- tryCatch(eval(expr, parent.frame()), error = function(e) {
        msg <- conditionMessage(e)
        if (grepl("missing", msg, fixed = FALSE)) {
          return(NULL)
        }
        stop(e)
      })
      if (!is.null(value)) {
        idx_list[[k]] <- value
      }
    }
  }

  if (length(dot_expr) == 1L) {
    idx_arg <- idx_list[[1]]
    if (.mlx_is_numeric_matrix_index(idx_arg, x$dim)) {
      return(.mlx_matrix_subset(x, idx_arg))
    }
    if (.mlx_is_numeric_matrix_shape(idx_arg)) {
      stop("Matrix index must have one column per dimension.", call. = FALSE)
    }

    flat_sel <- .mlx_coerce_flat_index(idx_arg, length(x))
    coord_mat <- if (length(flat_sel)) {
      arrayInd(flat_sel + 1L, .dim = x$dim)
    } else {
      matrix(integer(0), nrow = 0, ncol = length(x$dim))
    }
    return(.mlx_matrix_subset(x, coord_mat))
  }

  out <- x
  for (axis in seq_len(ndim)) {
    idx <- if (axis <= length(idx_list)) idx_list[[axis]] else NULL
    sel <- .normalize_index_vector(idx, out$dim[axis])
    if (is.null(sel)) next

    # If sel is an mlx array, pass its pointer; otherwise pass the R vector
    sel_arg <- if (is.mlx(sel)) sel$ptr else sel
    ptr <- cpp_mlx_take(out$ptr, sel_arg, axis - 1L)
    out <- .mlx_wrap_result(ptr, out$device)
  }

  if (drop) {
    keep <- out$dim != 1L
    if (!all(keep) && length(out$dim) > 0L) {
      new_dim <- out$dim[keep]
      ptr <- if (length(new_dim) == 0L) {
        out$ptr
      } else {
        cpp_mlx_reshape(out$ptr, as.integer(new_dim))
      }
      out <- .mlx_wrap_result(ptr, out$device)
      if (length(new_dim) == 0L) {
        out$dim <- integer(0)
      }
    }
  }

  out
}

#' @rdname mlx_subset
#' @method [<- mlx
#' @export
`[<-.mlx` <- function(x, ..., value) {
  stopifnot(is.mlx(x))
  ndim <- length(x$dim)
  if (ndim == 0L) {
    stop("Cannot assign to a scalar mlx array.", call. = FALSE)
  }

  dot_expr <- as.list(substitute(alist(...)))[-1]
  if (length(dot_expr) > ndim &&
      !(length(dot_expr) == 1L && .mlx_is_numeric_matrix_index(eval(dot_expr[[1]], parent.frame()), x$dim))) {
    stop("Incorrect number of indices supplied.", call. = FALSE)
  }

  idx_list <- vector("list", ndim)
  if (length(dot_expr)) {
    for (k in seq_along(dot_expr)) {
      expr <- dot_expr[[k]]
      value_idx <- tryCatch(eval(expr, parent.frame()), error = function(e) {
        msg <- conditionMessage(e)
        if (grepl("missing", msg, fixed = FALSE)) {
          return(NULL)
        }
        stop(e)
      })
      if (!is.null(value_idx)) {
        idx_list[[k]] <- value_idx
      }
    }
  }

  if (length(dot_expr) == 1L && .mlx_is_numeric_matrix_index(idx_list[[1]], x$dim)) {
    idx_mat <- idx_list[[1]]
    return(.mlx_matrix_assign(x, idx_mat, value))
  }

  if (length(dot_expr) == 1L) {
    if (.mlx_is_numeric_matrix_shape(idx_list[[1]])) {
      stop("Matrix index must have one column per dimension.", call. = FALSE)
    }
    flat_sel <- .mlx_coerce_flat_index(idx_list[[1]], length(x))
    if (!length(flat_sel)) {
      return(x)
    }

    coord_mat <- arrayInd(flat_sel + 1L, .dim = x$dim)
    return(.mlx_matrix_assign(x, coord_mat, value))
  }

  normalized <- vector("list", ndim)
  empty_selection <- FALSE
  for (axis in seq_len(ndim)) {
    idx <- if (axis <= length(idx_list)) idx_list[[axis]] else NULL
    sel <- .normalize_index_vector(idx, x$dim[axis])
    if (!is.null(sel) && length(sel) == 0L) {
      empty_selection <- TRUE
      break
    }
    normalized[axis] <- list(sel)
  }

  if (empty_selection) {
    return(x)
  }

  # Determine slice parameters when possible
  slice_params <- lapply(seq_len(ndim), function(axis) {
    sel <- normalized[[axis]]
    dim_len <- x$dim[axis]
    if (is.null(sel)) {
      list(all = TRUE, start = 0L, stop = dim_len, stride = 1L, len = dim_len)
    } else {
      stride <- if (length(sel) <= 1L) 1L else diff(sel)
      if (length(stride) > 1L && !all(stride == stride[1])) {
        return(NULL)
      }
      stride_val <- if (length(stride) == 0L) 1L else stride[1]
      list(
        all = FALSE,
        start = sel[1],
        stop = utils::tail(sel, 1) + stride_val,
        stride = stride_val,
        len = length(sel)
      )
    }
  })

  dims_sel <- vapply(seq_len(ndim), function(axis) {
    sel <- normalized[[axis]]
    if (is.null(sel)) x$dim[axis] else length(sel)
  }, integer(1))
  total_elems <- prod(dims_sel)
  if (total_elems == 0L) {
    return(x)
  }

  value_vec <- as.vector(value)
  if (length(value_vec) == 0L) {
    stop("Replacement value must have length >= 1.", call. = FALSE)
  }
  value_vec <- rep_len(value_vec, total_elems)
  value_array <- array(value_vec, dim = dims_sel)
  value_mlx <- as_mlx(value_array, dtype = x$dtype, device = x$device)

  if (!any(vapply(slice_params, is.null, logical(1)))) {
    start <- vapply(slice_params, `[[`, integer(1), "start")
    stop <- vapply(slice_params, `[[`, integer(1), "stop")
    strides <- vapply(slice_params, `[[`, integer(1), "stride")
    ptr <- cpp_mlx_slice_update(x$ptr, value_mlx$ptr, start, stop, strides)
    return(.mlx_wrap_result(ptr, x$device))
  }

  # Fallback to explicit element-wise assignment via index matrix
  full_indices <- lapply(seq_len(ndim), function(axis) {
    if (is.null(normalized[[axis]])) {
      seq.int(0L, x$dim[axis] - 1L)
    } else {
      normalized[[axis]]
    }
  })

  grid <- do.call(expand.grid, c(full_indices, KEEP.OUT.ATTRS = FALSE))
  coord_mat <- as.matrix(grid)
  if (!is.matrix(coord_mat)) {
    coord_mat <- matrix(coord_mat, ncol = ndim)
  }
  idx_mat <- coord_mat + 1L

  has_dupes <- nrow(idx_mat) > 1L && anyDuplicated(idx_mat)
  if (has_dupes) {
    base <- as.array(x)
    base[idx_mat] <- as.vector(value_array)
    return(as_mlx(base, dtype = x$dtype, device = x$device))
  }

  .mlx_matrix_assign(x, idx_mat, value_array)
}

#' Matrix-style subsetting helper.
#'
#' @param x `mlx` array to subset.
#' @param idx_mat Integer matrix of 1-based indices (rows correspond to points).
#' @return An `mlx` array containing the selected elements.
#' @noRd
.mlx_matrix_subset <- function(x, idx_mat) {
  idx_mat <- .mlx_coerce_index_matrix(idx_mat, x$dim, type = "subset")
  if (!nrow(idx_mat)) {
    flat <- mlx_flatten(x)
    res <- .mlx_wrap_result(cpp_mlx_take(flat$ptr, integer(0), 0L), x$device)
    res$dim <- integer(1)
    return(res)
  }

  linear_idx <- .mlx_linear_indices(idx_mat, x$dim)
  flat <- mlx_flatten(x)
  ptr <- cpp_mlx_take(flat$ptr, linear_idx, 0L)
  .mlx_wrap_result(ptr, x$device)
}

#' Matrix-style assignment helper.
#'
#' @param x `mlx` array to modify.
#' @param idx_mat Integer matrix of 1-based indices.
#' @param value Replacement values (recycled to match index count).
#' @return An `mlx` array with the assignments applied.
#' @noRd
.mlx_matrix_assign <- function(x, idx_mat, value) {
  idx_mat <- .mlx_coerce_index_matrix(idx_mat, x$dim, type = "assign")
  if (!nrow(idx_mat)) {
    return(x)
  }

  linear_idx <- .mlx_linear_indices(idx_mat, x$dim)
  total <- length(linear_idx)

  val_vec <- as.vector(value)
  if (!length(val_vec)) {
    stop("Replacement value must have length >= 1.", call. = FALSE)
  }
  val_vec <- rep_len(val_vec, total)

  value_mat <- matrix(val_vec, ncol = 1L)
  updates_mlx <- as_mlx(value_mat, dtype = x$dtype, device = x$device)
  idx_mlx <- as_mlx(linear_idx, dtype = "int64", device = x$device)

  flat <- mlx_flatten(x)
  flat_updated <- .mlx_scatter_axis(flat, idx_mlx, updates_mlx, axis = 0L)
  mlx_reshape(flat_updated, x$dim)
}

#' Compute linear indices from multi-axis coordinates.
#'
#' @param index_matrix Matrix of zero-based indices (rows = elements).
#' @param dim_sizes Integer vector of dimension sizes.
#' @return Integer vector of flattened indices.
#' @noRd
.mlx_linear_indices <- function(index_matrix, dim_sizes) {
  if (length(dim_sizes) == 0L) {
    return(integer(0))
  }
  if (!is.matrix(index_matrix)) {
    index_matrix <- matrix(index_matrix, ncol = length(dim_sizes))
  }
  strides <- vapply(seq_along(dim_sizes), function(k) {
    if (k == length(dim_sizes)) {
      1L
    } else {
      as.integer(prod(dim_sizes[(k + 1):length(dim_sizes)]))
    }
  }, integer(1))
  linear <- index_matrix %*% strides
  as.integer(linear)
}

#' Normalize subsetting index to 0-indexed integers
#' 
#' @param idx Logical, numeric, or NULL index vector.
#' @param dim_size Integer length of the dimension.
#' @return Integer vector (0-indexed) or NULL.
#' @noRd
.normalize_index_vector <- function(idx, dim_size) {
  if (is.null(idx)) {
    return(NULL)
  }

  if (is.mlx(idx)) {
    idx <- as.array(idx)
  }

  if (is.logical(idx)) {
    len <- length(idx)
    if (len == 0L) {
      return(integer(0))
    }
    if (len == 1L) {
      idx <- rep(idx, dim_size)
    } else if (len != dim_size) {
      stop("Logical index length must be 1 or match dimension length.", call. = FALSE)
    }
    idx <- which(idx)
  }

  if (is.numeric(idx)) {
    if (length(idx) == 0L) {
      return(integer(0))
    }
    if (any(is.na(idx))) {
      stop("Index contains NA values.", call. = FALSE)
    }
    if (any(idx == 0L)) {
      stop("Index contains zeros, which are not allowed.", call. = FALSE)
    }
    if (any(idx != floor(idx))) {
      stop("Numeric indices must be whole numbers.", call. = FALSE)
    }
    negative <- idx < 0
    if (any(negative)) {
      if (!all(negative)) {
        stop("Cannot mix positive and negative indices.", call. = FALSE)
      }
      idx_abs <- as.integer(abs(idx))
      if (any(idx_abs < 1L) || any(idx_abs > dim_size)) {
        stop("Index out of bounds.", call. = FALSE)
      }
      keep <- setdiff(seq_len(dim_size), unique(idx_abs))
      return(as.integer(keep - 1L))
    }

    idx <- as.integer(idx)
    if (any(idx < 1L) || any(idx > dim_size)) {
      stop("Index out of bounds.", call. = FALSE)
    }
    return(as.integer(idx - 1L))
  }

  stop("Unsupported index type.", call. = FALSE)
}

.mlx_is_numeric_matrix_index <- function(idx, dim_sizes) {
  ndim <- length(dim_sizes)
  if (is.null(idx)) {
    return(FALSE)
  }

  if (is.mlx(idx)) {
    return(isTRUE(length(idx$dim) >= 2L && tail(idx$dim, 1) == ndim) &&
             !identical(idx$dtype, "bool"))
  }

  is_array <- is.matrix(idx) || (is.array(idx) && length(dim(idx)) >= 2L)
  if (!is_array) {
    return(FALSE)
  }

  dims <- dim(idx)
  is.numeric(idx) && tail(dims, 1) == ndim
}

.mlx_is_numeric_matrix_shape <- function(idx) {
  if (is.null(idx)) {
    return(FALSE)
  }

  if (is.mlx(idx)) {
    return(!identical(idx$dtype, "bool") && length(idx$dim) >= 2L)
  }

  (is.matrix(idx) || (is.array(idx) && length(dim(idx)) >= 2L)) && is.numeric(idx)
}

.mlx_coerce_index_matrix <- function(idx, dim_sizes, type = c("subset", "assign")) {
  type <- match.arg(type)

  if (is.mlx(idx)) {
    mat <- as.array(idx)
  } else {
    mat <- idx
  }

  if (!is.numeric(mat)) {
    stop("Matrix index must be numeric.", call. = FALSE)
  }

  dims <- dim(mat)
  if (length(dims) < 2L) {
    stop("Matrix index must have at least two dimensions.", call. = FALSE)
  }
  if (tail(dims, 1) != length(dim_sizes)) {
    stop("Matrix index must have one column per dimension.", call. = FALSE)
  }

  n_points <- prod(dims[-length(dims)])
  dim(mat) <- c(n_points, length(dim_sizes))
  idx_mat <- mat

  if (n_points == 0L) {
    return(matrix(integer(0), nrow = 0, ncol = length(dim_sizes)))
  }

  if (any(is.na(idx_mat))) {
    stop("Index contains NA values.", call. = FALSE)
  }
  if (any(idx_mat < 1L)) {
    stop("Matrix indices must be positive (1-based).", call. = FALSE)
  }
  if (any(idx_mat != floor(idx_mat))) {
    stop("Matrix indices must be whole numbers.", call. = FALSE)
  }

  dim_sizes <- as.integer(dim_sizes)
  for (axis in seq_len(length(dim_sizes))) {
    col <- idx_mat[, axis]
    if (any(col > dim_sizes[axis])) {
      stop("Index out of bounds.", call. = FALSE)
    }
    idx_mat[, axis] <- as.integer(col - 1L)
  }

  storage.mode(idx_mat) <- "integer"
  idx_mat
}

.mlx_coerce_flat_index <- function(idx, total_len) {
  if (is.null(idx)) {
    return(integer(0))
  }

  if (is.mlx(idx)) {
    idx <- as.array(idx)
  }

  if (is.logical(idx)) {
    len <- length(idx)
    if (len == 0L) {
      return(integer(0))
    }
    if (len == 1L) {
      idx <- rep(idx, total_len)
    } else if (len != total_len) {
      stop("Logical index length must be 1 or match the number of elements.", call. = FALSE)
    }
    pos <- which(idx)
    return(as.integer(pos - 1L))
  }

  if (is.numeric(idx)) {
    if (!length(idx)) {
      return(integer(0))
    }
    if (any(is.na(idx))) {
      stop("Index contains NA values.", call. = FALSE)
    }
    if (any(idx == 0L)) {
      stop("Index contains zeros, which are not allowed.", call. = FALSE)
    }
    if (any(idx != floor(idx))) {
      stop("Numeric indices must be whole numbers.", call. = FALSE)
    }
    negative <- idx < 0
    if (any(negative)) {
      if (!all(negative)) {
        stop("Cannot mix positive and negative indices.", call. = FALSE)
      }
      idx_abs <- as.integer(abs(idx))
      if (any(idx_abs < 1L) || any(idx_abs > total_len)) {
        stop("Index out of bounds.", call. = FALSE)
      }
      keep <- setdiff(seq_len(total_len), unique(idx_abs))
      return(as.integer(keep - 1L))
    }

    idx <- as.integer(idx)
    if (any(idx > total_len)) {
      stop("Index out of bounds.", call. = FALSE)
    }
    return(as.integer(idx - 1L))
  }

  stop("Unsupported index type.", call. = FALSE)
}
