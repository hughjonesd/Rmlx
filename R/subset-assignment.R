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
      return(.mlx_assign_matrix(x, resolved$coord, value))
    }
  }

  # Convert all non-NULL indices to mlx (keeps dtype for logical/boolean)
  idx_mlx <- lapply(idx_list, function(idx) {
    if (is.null(idx)) return(NULL)
    as_mlx(idx, device = x$device)
  })

  all_bool <- all(vapply(idx_mlx, function(idx) {
    is.null(idx) || identical(mlx_dtype(idx), "bool")
  }, logical(1)))

  if (all_bool) {
    # Replace NULL with all-TRUE mask and use boolean masked assignment
    masks <- lapply(seq_len(ndim), function(i) {
      if (is.null(idx_mlx[[i]])) {
        mlx_ones(shape[i], dtype = "bool", device = x$device)
      } else {
        idx_mlx[[i]]
      }
    })
    return(.mlx_assign_boolean_mask(x, masks, shape, value))
  }

  .mlx_assign_numeric(x, idx_mlx, shape, value)
}

#' Scatter-style assignment helper
#'
#' Performs the equivalent of `x[indices[[1]], indices[[2]], ...] <- value`
#' using MLX `scatter()`. Indices are numeric vectors (1-based, like R).
#'
#' @param x `mlx` array to update.
#' @param indices List of numeric vectors, one per axis of `x`.
#' @param value Replacement values (recycled like base R).
#' @return Updated `mlx` array.
#' @noRd
scatter_assign <- function(x, indices, value) {
  stopifnot(is_mlx(x))
  shape <- mlx_shape(x)
  ndim <- length(shape)
  if (length(indices) != ndim) {
    stop("length(indices) must match rank of x", call. = FALSE)
  }

  # Normalize and validate indices (1-based to 0-based)
  idx_norm <- lapply(seq_len(ndim), function(i) {
    idx <- as.integer(indices[[i]])
    if (any(is.na(idx))) stop("indices must be finite integers", call. = FALSE)
    if (any(idx < 1L) || any(idx > shape[i])) {
      stop("indices out of bounds for dimension ", i, call. = FALSE)
    }
    idx - 1L
  })

  lens <- vapply(idx_norm, length, integer(1))
  if (any(lens == 0L)) return(x)

  # Prepare updates in row-major order expected by scatter
  updates_r <- array(value, dim = c(lens, rep(1L, ndim)))
  updates <- as_mlx(updates_r, dtype = mlx_dtype(x), device = x$device)

  idx_vecs <- lapply(idx_norm, function(v) as_mlx(v, dtype = "int64", device = x$device))
  idx_grid <- mlx_meshgrid(idx_vecs, sparse = FALSE, indexing = "ij", device = x$device)

  axes <- seq_len(ndim) - 1L
  ptr <- cpp_mlx_scatter(x$ptr, idx_grid, updates$ptr, axes, x$device)
  new_mlx(ptr, x$device)
}

.mlx_assign_numeric <- function(x, idx_mlx, shape, value) {
  ndim <- length(shape)
  normalized <- vector("list", ndim)
  dims_sel <- integer(ndim)
  empty <- FALSE

  for (axis in seq_len(ndim)) {
    dim_len <- shape[axis]
    idx <- if (axis <= length(idx_mlx)) idx_mlx[[axis]] else NULL
    pos <- if (is.null(idx)) {
      seq_len(dim_len)
    } else if (identical(mlx_dtype(idx), "bool")) {
      which(as.array(idx))
    } else {
      as.integer(as.array(idx))
    }

    pos <- .resolve_to_positive_indices(pos, dim_len)
    if (length(pos) == 0L) {
      normalized[[axis]] <- pos
      dims_sel[axis] <- 0L
      empty <- TRUE
      break
    }
    if (anyDuplicated(pos) > 0L) {
      stop("Duplicate indices are not allowed in assignment.", call. = FALSE)
    }

    normalized[[axis]] <- as.integer(pos - 1L)
    dims_sel[axis] <- length(pos)
  }

  if (empty) {
    return(x)
  }

  # Delegate to scatter_assign (expects 1-based indices), re-add 1 to normalized
  scatter_assign(x, lapply(normalized, `+`, 1L), value)
}


# Flatten an mlx array in R's column-major order
.mlx_flatten_r_order <- function(x) {
  if (length(mlx_shape(x)) <= 1L) {
    return(mlx_reshape(x, c(length(x))))
  }
  tx <- t(x)
  tx <- mlx_contiguous(tx)
  mlx_reshape(tx, c(length(x)))
}

# Prepare updates flattened in R (column-major) order for a selection
.mlx_prepare_updates_for_selection <- function(value, target_len, dtype, device) {
  value_mlx <- .mlx_cast(as_mlx(value), dtype = dtype, device = device)
  value_len <- length(value_mlx)
  if (value_len == 0L) {
    stop("Replacement value must have length >= 1.", call. = FALSE)
  }
  if (value_len != 1L && target_len %% value_len != 0L) {
    stop("Number of items to replace is not a multiple of replacement length", call. = FALSE)
  }
  tiles <- target_len %/% value_len
  flat <- .mlx_flatten_r_order(value_mlx)
  mlx_tile(flat, tiles)
}

# Boolean mask assignment helper using masked_scatter
.mlx_assign_boolean_mask <- function(x, idx_list, shape, value) {
  ndim <- length(shape)

  # Reshape each mask to have singleton dimensions in all axes except its own
  reshaped_masks <- lapply(seq_len(ndim), function(i) {
    new_shape <- rep(1L, ndim)
    new_shape[i] <- shape[i]
    mlx_reshape(idx_list[[i]], new_shape)
  })

  # Broadcast all masks to the same shape
  broadcasted <- mlx_broadcast_arrays(reshaped_masks, device = x$device)
  # Combine with logical AND
  combined_mask <- mlx_stack(broadcasted)
  combined_mask <- mlx_all(combined_mask, axes = 1)
  if (! any(combined_mask)) {
    return(x)
  }

  # Count selected elements and prepare updates in R (column-major) order
  n_selected <- as.integer(mlx_sum(combined_mask))
  x_dtype <- mlx_dtype(x)
  updates <- .mlx_prepare_updates_for_selection(value, n_selected, x_dtype, x$device)

  ptr <- cpp_mlx_masked_scatter_colmajor(x$ptr, combined_mask$ptr, updates$ptr, x$device)
  new_mlx(ptr, x$device)
}

# Matrix-style assignment helper.
.mlx_assign_matrix <- function(x, idx_mat, value) {
  dims <- mlx_shape(x)
  idx_mat <- .mlx_coerce_index_matrix(idx_mat, dims, type = "assign")
  if (!nrow(idx_mat)) {
    return(x)
  }
  x_dtype <- mlx_dtype(x)

  linear_idx <- .mlx_linear_indices(idx_mat, dims)
  val_vec <- as.vector(value)
  if (!length(val_vec)) {
    stop("Replacement value must have length >= 1.", call. = FALSE)
  }
  if (anyDuplicated(linear_idx)) {
    stop("Duplicate indices are not allowed in assignment.", call. = FALSE)
  }
  val_vec <- rep_len(val_vec, length(linear_idx))

  # Scatter on flattened array using row-major linear indices
  flat <- mlx_flatten(x)
  updated <- scatter_assign(flat, list(linear_idx + 1L), val_vec)
  mlx_reshape(updated, dims)
}
