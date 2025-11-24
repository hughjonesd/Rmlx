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

  # Reject NA indices early to match documented behavior
  if (any(vapply(idx_list, function(i) any(is.na(i)), logical(1)))) {
    stop("Index contains NA values.", call. = FALSE)
  }

  # Matrix/array indexing (one coordinate per row) delegates to helper
  if (length(dot_expr) == 1L && ndim > 1L) {
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

  # Caller (.mlx_assign_numeric) already validated bounds/duplicates.
  # Convert 1-based indices to 0-based, staying in MLX when possible.
  idx_norm <- lapply(indices, function(idx) {
    as_mlx(idx - 1L, dtype = "int64", device = x$device)
  })

  lens <- vapply(idx_norm, length, integer(1))
  if (any(lens == 0L)) return(x)

  # Prepare updates using existing tiling helper to avoid R materialisation
  target_len <- prod(lens)
  value_mlx <- as_mlx(value, dtype = mlx_dtype(x), device = x$device)
  val_len <- length(value_mlx)
  .check_value_fits(val_len, target_len)
  tiles <- target_len %/% val_len
  flat <- .mlx_flatten_r_order(value_mlx)
  updates_flat <- if (tiles == 1L) flat else mlx_tile(flat, tiles)
  rev_shape <- rev(c(lens, rep(1L, ndim)))
  updates_rev <- mlx_reshape(updates_flat, rev_shape)
  updates <- new_mlx(cpp_mlx_transpose(updates_rev$ptr), x$device)

  idx_grid <- mlx_meshgrid(idx_norm, sparse = FALSE, indexing = "ij", device = x$device)

  axes <- seq_len(ndim) - 1L
  ptr <- cpp_mlx_scatter(x$ptr, idx_grid, updates$ptr, axes, x$device)
  new_mlx(ptr, x$device)
}

.check_value_fits <- function(val_len, target_len) {
  if (val_len == 0L) {
    stop("Replacement value must have length >= 1.", call. = FALSE)
  }
  if (target_len %% val_len != 0L) {
    stop("Number of items to replace is not a multiple of replacement length", call. = FALSE)
  }
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
  ptr <- cpp_mlx_flatten_r_order(x$ptr)
  out <- new_mlx(ptr, x$device)
  mlx_reshape(out, c(length(x)))
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
    # nothing to replace
    return(x)
  }

  # Count selected elements and prepare updates in R (column-major) order
  n_selected <- as.integer(mlx_sum(combined_mask))
  value <- .mlx_cast(as_mlx(value), dtype = mlx_dtype(x), device = mlx_device(x))
  value_len <- length(value)
  .check_value_fits(value_len, n_selected)
  tiles <- n_selected %/% value_len
  value <- .mlx_flatten_r_order(value)
  value <- mlx_tile(value, tiles)

  ptr <- cpp_mlx_masked_scatter(x$ptr, combined_mask$ptr, value$ptr, x$device)
  new_mlx(ptr, x$device)
}

# Matrix-style assignment helper.
.mlx_assign_matrix <- function(x, idx_mat, value) {
  dims <- mlx_shape(x)
  idx_mat <- .mlx_check_index_matrix(idx_mat, dims)
  idx_mat <- idx_mat - 1L
  idx_mat <- as_mlx(idx_mat, dtype = "int64", device = mlx_device(x))
  if (!nrow(idx_mat)) {
    return(x)
  }
  x_dtype <- mlx_dtype(x)

  .check_value_fits(length(value), nrow(idx_mat))
  if (.duplicated_rows_lex(idx_mat)) {
    stop("Duplicate indices are not allowed in assignment.", call. = FALSE)
  }

  value <- mlx_repeat(value, nrow(idx_mat) %/% length(value))

  # Per-axis indices (0-based) as mlx arrays
  idx_list <- mlx_split(idx_mat, sections = ncol(idx_mat), axis = 2L)
  idx_list <- lapply(idx_list, drop)
  value <- .mlx_cast(value, dtype = x_dtype, device = mlx_device(x))
  value <- mlx_reshape(value, c(nrow(idx_mat), rep(1L, length(dims))))
  axes <- seq_len(length(dims)) - 1L

  ptr <- cpp_mlx_scatter(x$ptr, idx_list, value$ptr, axes, mlx_device(x))
  new_mlx(ptr, mlx_device(x))
}
