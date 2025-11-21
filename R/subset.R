
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
#' @param value Replacement values, recycled to match the selection.
#' @return The subsetted MLX object.
#' @seealso [mlx.core.take](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.take)
#' @name mlx_subset
#' @export
#' @examples
#' x <- mlx_matrix(1:9, 3, 3)
#' x[1, ]
`[.mlx` <- function(x, ..., drop = FALSE) {
  shape <- mlx_shape(x)
  ndim <- length(shape)
  if (ndim == 0L) {
    stop("Cannot subset a scalar mlx array.", call. = FALSE)
  }

  # Evaluate and collect index arguments supplied through ...
  dot_expr <- as.list(substitute(alist(...)))[-1]
  idx_list <- .mlx_collect_indices(dot_expr, ndim, parent.frame())

  # Handle matrix-style coordinates or flattened vector indices via dedicated helper
  if (length(dot_expr) == 1L) {
    resolved <- .mlx_resolve_single_index(idx_list[[1]], shape)
    if (!is.null(resolved)) {
      return(.mlx_matrix_subset(x, resolved$coord))
    }
  }

  # Apply take() sequentially across axes to realise the remaining selection
  out <- x
  for (axis in seq_len(ndim)) {
    current_shape <- mlx_shape(out)
    idx <- if (axis <= length(idx_list)) idx_list[[axis]] else NULL
    sel <- .normalize_index_vector(idx, current_shape[axis])
    if (is.null(sel)) next

    # If sel is an mlx array, pass its pointer; otherwise pass the R vector
    sel_arg <- if (is_mlx(sel)) sel$ptr else sel
    ptr <- cpp_mlx_take(out$ptr, sel_arg, axis - 1L)
    out <- new_mlx(ptr, out$device)
  }

  # Optionally drop singleton dimensions (default is FALSE to mirror package conventions)
  if (drop) {
    shape_out <- mlx_shape(out)
    keep <- shape_out != 1L
    if (!all(keep) && length(shape_out) > 0L) {
      new_dim <- shape_out[keep]
      ptr <- if (length(new_dim) == 0L) {
        out$ptr
      } else {
        cpp_mlx_reshape(out$ptr, as.integer(new_dim))
      }
      out <- new_mlx(ptr, out$device)
      if (length(new_dim) == 0L) {
        dim(out) <- integer(0)
      }
    }
  }

  out
}

#' Boolean mask assignment helper using masked_scatter
#'
#' @param x `mlx` array to modify.
#' @param idx_list List of per-axis indices.
#' @param shape Integer vector of dimension sizes.
#' @param value Replacement values.
#' @return An `mlx` array with the assignments applied.
#' @noRd
.mlx_assign_boolean_mask <- function(x, idx_list, shape, value) {
  ndim <- length(shape)

  # Convert all indices to boolean masks for uniform handling
  masks <- lapply(seq_len(ndim), function(i) {
    idx <- if (i <= length(idx_list)) idx_list[[i]] else NULL

    if (is.null(idx)) {
      # NULL means select all - create all TRUE mask
      mlx_ones(shape[i], dtype = "bool", device = x$device)
    } else if (is_mlx(idx) && mlx_dtype(idx) == "bool") {
      # Already a boolean mlx mask - use as-is without materialization
      idx
    } else {
      # Integer or R logical indices - convert to boolean mask
      if (is_mlx(idx)) {
        # Keep as mlx and convert to boolean on GPU
        if (mlx_length(idx) == 0L) {
          mlx_zeros(shape[i], dtype = "bool", device = x$device)
        } else {
          # Check if negative by looking at first element
          first_val <- as.numeric(mlx_take(idx, 0L, axis = 0L))
          if (first_val < 0) {
            # Negative: exclude these positions
            # Create all-TRUE mask then set abs(idx) positions to FALSE
            mask <- mlx_ones(shape[i], dtype = "bool", device = x$device)
            # Use scatter to set positions to FALSE
            # Convert to 0-based indices
            positions <- mlx_abs(idx) - 1L
            updates <- mlx_zeros(mlx_length(positions), dtype = "bool", device = x$device)
            .mlx_scatter_axis(mask, positions, updates, axis = 0L)
          } else {
            # Positive: include only these positions using broadcasting
            # positions = [0, 1, 2, ..., shape[i]-1]
            positions <- mlx_arange(shape[i], device = x$device)
            # Convert idx to 0-based
            idx_0based <- idx - 1L
            # Reshape for broadcasting: positions (N, 1), idx (1, M)
            pos_col <- mlx_reshape(positions, c(shape[i], 1L))
            idx_row <- mlx_reshape(idx_0based, c(1L, mlx_length(idx)))
            # Compare and reduce: any position matches any idx value
            matches <- pos_col == idx_row
            mlx_any(matches, axis = 2L)
          }
        }
      } else if (is.logical(idx)) {
        # R logical - expand if needed and convert directly to mlx boolean
        len <- length(idx)
        if (len == 1L) {
          idx <- rep(idx, shape[i])
        } else if (len != shape[i]) {
          stop("Logical index length must be 1 or match dimension length.", call. = FALSE)
        }
        as_mlx(idx, dtype = "bool", device = x$device)
      } else {
        # R integer indices - convert to boolean mask
        if (length(idx) == 0L) {
          mlx_zeros(shape[i], dtype = "bool", device = x$device)
        } else if (any(idx < 0)) {
          # Negative: exclude these positions
          if (!all(idx < 0)) {
            stop("Cannot mix positive and negative indices.", call. = FALSE)
          }
          mask_r <- rep(TRUE, shape[i])
          mask_r[abs(idx)] <- FALSE
          as_mlx(mask_r, dtype = "bool", device = x$device)
        } else {
          # Positive: include these positions
          mask_r <- logical(shape[i])
          mask_r[idx] <- TRUE
          as_mlx(mask_r, dtype = "bool", device = x$device)
        }
      }
    }
  })

  # Combine masks with broadcasting
  if (ndim == 1) {
    combined_mask <- masks[[1]]
  } else {
    # Reshape each mask to have singleton dimensions in all axes except its own
    reshaped_masks <- lapply(seq_len(ndim), function(i) {
      new_shape <- rep(1L, ndim)
      new_shape[i] <- shape[i]
      mlx_reshape(masks[[i]], new_shape)
    })

    # Broadcast all masks to the same shape
    broadcasted <- mlx_broadcast_arrays(reshaped_masks, device = x$device)

    # Combine with logical AND
    combined_mask <- broadcasted[[1]]
    for (i in 2:ndim) {
      combined_mask <- combined_mask & broadcasted[[i]]
    }
  }

  # Count selected elements and prepare updates
  n_selected <- as.integer(mlx_sum(combined_mask))
  if (n_selected == 0L) {
    return(x)
  }

  x_dtype <- mlx_dtype(x)

  # Convert R value to mlx if needed, ensure correct dtype
  if (!is_mlx(value)) {
    value <- mlx_vector(as.vector(value), dtype = x_dtype, device = x$device)
  } else {
    value <- .mlx_cast(value, dtype = x_dtype, device = x$device)
  }

  # Prepare updates - handle recycling on GPU
  value_len <- length(value)
  if (value_len == 0L) {
    stop("Replacement value must have length >= 1.", call. = FALSE)
  }

  # Check that n_selected is a multiple of value_len (R's recycling rule)
  if (value_len != 1L && n_selected %% value_len != 0L) {
    stop("number of items to replace is not a multiple of replacement length", call. = FALSE)
  }

  if (value_len == n_selected) {
    updates <- mlx_flatten(value)
  } else if (value_len == 1L) {
    # Broadcast scalar to needed length
    updates <- mlx_broadcast_to(value, n_selected, device = x$device)
  } else {
    # Recycle: tile to exact number of repeats needed
    n_tiles <- n_selected %/% value_len
    value_flat <- mlx_flatten(value)
    updates <- mlx_tile(value_flat, n_tiles)
  }

  # MLX uses row-major, R uses column-major
  # transpose() reverses all axes, converting between the two orderings
  x_t <- t(x)
  mask_t <- t(combined_mask)

  ptr_t <- cpp_mlx_masked_scatter(x_t$ptr, mask_t$ptr, updates$ptr, x$device)
  result_t <- new_mlx(ptr_t, x$device)

  # Transpose back
  t(result_t)
}

#' Integer index assignment helper
#'
#' @param x `mlx` array to modify.
#' @param idx_list List of per-axis indices.
#' @param shape Integer vector of dimension sizes.
#' @param value Replacement values.
#' @return An `mlx` array with the assignments applied.
#' @noRd
.mlx_assign_indices <- function(x, idx_list, shape, value) {
  ndim <- length(shape)

  prep <- .mlx_prepare_assignment_indices(idx_list, shape)
  if (prep$empty) {
    return(x)
  }

  dims_sel <- prep$dims_sel
  total_elems <- prod(dims_sel)
  stopifnot(total_elems >= 0L)
  if (total_elems == 0L) return(x)

  value_vec <- as.vector(value)
  if (length(value_vec) == 0L) {
    stop("Replacement value must have length >= 1.", call. = FALSE)
  }
  value_vec <- rep_len(value_vec, total_elems)

  # Store values for both slice (array) and scatter (vector) paths
  value_array <- array(value_vec, dim = dims_sel)
  x_dtype <- mlx_dtype(x)
  value_mlx_tensor <- as_mlx(value_array, dtype = x_dtype, device = x$device)

  value_mlx_vec <- mlx_vector(value_vec, dtype = x_dtype, device = x$device)

  slice <- .mlx_slice_parameters(prep$normalized, shape)
  if (slice$can_slice) {
    ptr <- cpp_mlx_slice_update(x$ptr, value_mlx_tensor$ptr, slice$start, slice$stop, slice$stride)
    return(new_mlx(ptr, x$device))
  }

  has_dupes <- any(vapply(prep$normalized, function(sel) {
    !is.null(sel) && length(sel) > 1L && anyDuplicated(sel)
  }, logical(1)))

  if (has_dupes) {
    full_indices <- Map(function(sel, dim_len) {
      if (is.null(sel)) {
        if (dim_len == 0L) integer(0) else seq.int(0L, dim_len - 1L)
      } else {
        sel
      }
    }, prep$normalized, shape)
    grid <- do.call(expand.grid, c(full_indices, KEEP.OUT.ATTRS = FALSE))
    coord_mat <- as.matrix(grid)
    if (!is.matrix(coord_mat)) {
      coord_mat <- matrix(coord_mat, ncol = ndim)
    }
    idx_mat <- coord_mat + 1L

    base <- as.array(x)
    base[idx_mat] <- value_vec
    return(as_mlx(base, dtype = x_dtype, device = x$device))
  }

  normalized_int <- lapply(prep$normalized, function(sel) {
    if (is.null(sel)) NULL else as.integer(sel)
  })

  ptr <- cpp_mlx_assign(x$ptr, normalized_int, value_mlx_vec$ptr, as.integer(shape))
  new_mlx(ptr, x$device)
}

#' @rdname mlx_subset
#' @export
`[<-.mlx` <- function(x, ..., value) {
  stopifnot(is_mlx(x))
  shape <- mlx_shape(x)
  ndim <- length(shape)
  if (ndim == 0L) {
    stop("Cannot assign to a scalar mlx array.", call. = FALSE)
  }

  # Evaluate index arguments; allow fewer than ndim entries when trailing dimensions are implied
  dot_expr <- as.list(substitute(alist(...)))[-1]
  idx_list <- .mlx_collect_indices(dot_expr, ndim, parent.frame())

  # Matrix/array indexing (one coordinate per row) delegates to helper
  if (length(dot_expr) == 1L) {
    resolved <- .mlx_resolve_single_index(idx_list[[1]], shape)
    if (!is.null(resolved)) {
      return(.mlx_matrix_assign(x, resolved$coord, value))
    }
  }

  # Check if any indices are boolean/logical (mlx or R)
  has_bool_mask <- any(vapply(idx_list, function(idx) {
    if (is.null(idx)) return(FALSE)
    if (is_mlx(idx)) return(mlx_dtype(idx) == "bool")
    return(is.logical(idx))
  }, logical(1)))

  if (has_bool_mask) {
    .mlx_assign_boolean_mask(x, idx_list, shape, value)
  } else {
    .mlx_assign_indices(x, idx_list, shape, value)
  }
}

#' Matrix-style subsetting helper.
#'
#' @param x `mlx` array to subset.
#' @param idx_mat Integer matrix of 1-based indices (rows correspond to points).
#' @return An `mlx` array containing the selected elements.
#' @noRd
.mlx_matrix_subset <- function(x, idx_mat) {
  dims <- mlx_shape(x)
  idx_mat <- .mlx_coerce_index_matrix(idx_mat, dims, type = "subset")
  if (!nrow(idx_mat)) {
    flat <- mlx_flatten(x)
    res <- new_mlx(cpp_mlx_take(flat$ptr, integer(0), 0L), x$device)
    dim(res) <- integer(1)
    return(res)
  }

  linear_idx <- .mlx_linear_indices(idx_mat, dims)
  flat <- mlx_flatten(x)
  ptr <- cpp_mlx_take(flat$ptr, linear_idx, 0L)
  new_mlx(ptr, x$device)
}

#' Matrix-style assignment helper.
#'
#' @param x `mlx` array to modify.
#' @param idx_mat Integer matrix of 1-based indices.
#' @param value Replacement values (recycled to match index count).
#' @return An `mlx` array with the assignments applied.
#' @noRd
.mlx_matrix_assign <- function(x, idx_mat, value) {
  dims <- mlx_shape(x)
  idx_mat <- .mlx_coerce_index_matrix(idx_mat, dims, type = "assign")
  if (!nrow(idx_mat)) {
    return(x)
  }
  x_dtype <- mlx_dtype(x)

  linear_idx <- .mlx_linear_indices(idx_mat, dims)
  total <- length(linear_idx)

  val_vec <- as.vector(value)
  if (!length(val_vec)) {
    stop("Replacement value must have length >= 1.", call. = FALSE)
  }
  val_vec <- rep_len(val_vec, total)

  idx_mlx <- as_mlx(linear_idx, dtype = "int64", device = x$device)
  idx_rank <- length(mlx_shape(idx_mlx))
  total_len <- as.integer(total)
  updates_dim <- c(total_len, rep.int(1L, idx_rank))
  updates_mlx <- mlx_array(
    val_vec,
    dim = updates_dim,
    dtype = x_dtype,
    device = x$device
  )

  flat <- mlx_flatten(x)
  flat_updated <- .mlx_scatter_axis(flat, idx_mlx, updates_mlx, axis = 0L)
  mlx_reshape(flat_updated, dims)
}

#' Evaluate and align index expressions with dimension count
#'
#' @param dot_expr List of unevaluated index expressions from `...`.
#' @param ndim Number of dimensions expected for the target array.
#' @param env Environment in which to evaluate the expressions.
#' @return List of length `ndim` containing evaluated indices (with `NULL`
#'   placeholders for omitted axes).
#' @noRd
.mlx_collect_indices <- function(dot_expr, ndim, env) {
  if (!length(dot_expr)) {
    return(vector("list", ndim))
  }

  evaluated <- lapply(dot_expr, function(expr) {
    tryCatch(
      eval(expr, env),
      error = function(e) {
        msg <- conditionMessage(e)
        if (grepl("missing", msg, fixed = FALSE)) {
          return(NULL)
        }
        stop(e)
      }
    )
  })

  if (length(evaluated) > ndim) {
    stop("Incorrect number of indices supplied.", call. = FALSE)
  }

  idx_list <- vector("list", ndim)
  if (length(evaluated)) {
    for (k in seq_along(evaluated)) {
      val <- evaluated[[k]]
      if (!is.null(val)) {
        idx_list[[k]] <- val
      }
    }
  }

  idx_list
}

#' Resolve a single (possibly matrix) index into coordinate rows
#'
#' @param idx Index object supplied by the user (may be `NULL`).
#' @param dim_sizes Integer vector of target dimension sizes.
#' @return `NULL` if the index cannot be handled specially, otherwise a list
#'   containing `coord`, a 1-based integer matrix of coordinates.
#' @noRd
.mlx_resolve_single_index <- function(idx, dim_sizes) {
  if (length(dim_sizes) == 0L) {
    return(NULL)
  }

  if (is.null(idx)) {
    coord_mat <- matrix(integer(0), nrow = 0, ncol = length(dim_sizes))
    return(list(coord = coord_mat))
  }

  if (.mlx_is_numeric_matrix_index(idx, dim_sizes)) {
    coord_mat <- idx
  } else {
    if (.mlx_is_numeric_matrix_shape(idx)) {
      stop("Matrix index must have one column per dimension.", call. = FALSE)
    }
    flat_sel <- .mlx_coerce_flat_index(idx, prod(dim_sizes))
    coord_mat <- if (length(flat_sel)) {
      arrayInd(flat_sel + 1L, .dim = dim_sizes)
    } else {
      matrix(integer(0), nrow = 0, ncol = length(dim_sizes))
    }
  }

  list(coord = coord_mat)
}

#' Normalise indices for assignment logic
#'
#' @param idx_list List of per-axis indices (possibly shorter than `dim_sizes`).
#' @param dim_sizes Integer vector of dimension sizes.
#' @return List with elements `normalized` (0-based indices or `NULL`),
#'   `dims_sel` (sizes of the selected region per axis), and `empty`
#'   indicating whether any axis selects zero elements.
#' @noRd
.mlx_prepare_assignment_indices <- function(idx_list, dim_sizes) {
  ndim <- length(dim_sizes)
  normalized <- vector("list", ndim)
  dims_sel <- dim_sizes
  empty <- FALSE

  for (axis in seq_len(ndim)) {
    idx <- if (axis <= length(idx_list)) idx_list[[axis]] else NULL
    sel <- .normalize_index_vector(idx, dim_sizes[axis])

    if (!is.null(sel) && length(sel) == 0L) {
      normalized[axis] <- list(sel)
      dims_sel[axis] <- 0L
      empty <- TRUE
      break
    }

    normalized[axis] <- list(sel)
    if (!is.null(sel)) {
      dims_sel[axis] <- length(sel)
    }
  }

  list(normalized = normalized, dims_sel = dims_sel, empty = empty)
}

#' Compute slice parameters for contiguous selections
#'
#' @param normalized List of normalised (0-based) indices per axis.
#' @param dim_sizes Integer vector of dimension sizes.
#' @return List with `can_slice` flag plus integer vectors `start`, `stop`,
#'   and `stride`.
#' @noRd
.mlx_slice_parameters <- function(normalized, dim_sizes) {
  ndim <- length(dim_sizes)
  start <- integer(ndim)
  stop <- integer(ndim)
  stride <- rep(1L, ndim)
  can_slice <- TRUE

  for (axis in seq_len(ndim)) {
    sel <- normalized[[axis]]
    dim_len <- dim_sizes[axis]
    if (is.null(sel)) {
      start[axis] <- 0L
      stop[axis] <- dim_len
      stride[axis] <- 1L
      next
    }

    diffs <- if (length(sel) <= 1L) integer(0) else diff(sel)
    if (length(diffs) > 1L && !all(diffs == diffs[1L])) {
      can_slice <- FALSE
    }
    stride_val <- if (length(diffs) == 0L) 1L else diffs[1L]
    if (stride_val <= 0L) {
      can_slice <- FALSE
    }
    start[axis] <- sel[1L]
    stride[axis] <- stride_val
    stop[axis] <- sel[length(sel)] + stride_val
  }

  list(
    can_slice = can_slice,
    start = start,
    stop = stop,
    stride = stride
  )
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

  if (is_mlx(idx)) {
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

#' Check whether an index represents matrix-style coordinates
#'
#' @param idx Candidate index (R matrix/array or `mlx` array).
#' @param dim_sizes Integer vector of target dimension sizes.
#' @return `TRUE` if `idx` encodes coordinate rows, `FALSE` otherwise.
#' @noRd
.mlx_is_numeric_matrix_index <- function(idx, dim_sizes) {
  ndim <- length(dim_sizes)
  if (is.null(idx)) {
    return(FALSE)
  }

  if (is_mlx(idx)) {
    dims <- mlx_shape(idx)
    return(isTRUE(length(dims) >= 2L && dims[length(dims)] == ndim) &&
             !identical(mlx_dtype(idx), "bool"))
  }

  dims <- dim(idx)

  is.array(idx) && is.numeric(idx) && dims[length(dims)] == ndim
}

#' Check whether an index has matrix/array shape (without validating columns)
#'
#' @param idx Candidate matrix/array index.
#' @return `TRUE` if `idx` is numeric with at least two dimensions.
#' @noRd
.mlx_is_numeric_matrix_shape <- function(idx) {
  if (is.null(idx)) {
    return(FALSE)
  }

  if (is_mlx(idx)) {
    return(!identical(mlx_dtype(idx), "bool") && length(dim(idx)) >= 2L)
  }

  (is.matrix(idx) || (is.array(idx) && length(dim(idx)) >= 2L)) && is.numeric(idx)
}

#' Coerce a matrix/array of coordinates into zero-based integer rows
#'
#' @param idx Numeric matrix/array or `mlx` array containing coordinates.
#' @param dim_sizes Integer vector of dimension sizes.
#' @param type Operation context (subset vs assign) used for error messages.
#' @return Integer matrix with one column per dimension, entries zero-based.
#' @noRd
.mlx_coerce_index_matrix <- function(idx, dim_sizes, type = c("subset", "assign")) {
  type <- match.arg(type)

  if (is_mlx(idx)) {
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
  if (dims[length(dims)] != length(dim_sizes)) {
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

#' Coerce 1D indices into zero-based integer positions
#'
#' @param idx Logical, numeric, or `mlx` vector index.
#' @param total_len Total length of the flattened array.
#' @return Integer vector of zero-based positions.
#' @noRd
.mlx_coerce_flat_index <- function(idx, total_len) {
  if (is.null(idx)) {
    return(integer(0))
  }

  if (is_mlx(idx)) {
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
