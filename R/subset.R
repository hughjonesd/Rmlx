
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

#' Resolve indices to positive 1-based integers
#'
#' Validates and normalizes R integer or logical indices, handling negative
#' indices by converting to their complement set. Returns 1-based positive
#' integer indices suitable for R indexing or conversion to boolean masks.
#'
#' @param idx Index vector (integer, logical, or mlx array to materialize).
#' @param dim_size Integer size of the dimension being indexed.
#' @return Integer vector of 1-based positive indices, or `NULL` if `idx` is
#'   `NULL`, or `integer(0)` if empty.
#' @noRd
.resolve_to_positive_indices <- function(idx, dim_size) {
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
      return(as.integer(keep))
    }

    idx <- as.integer(idx)
    if (any(idx < 1L) || any(idx > dim_size)) {
      stop("Index out of bounds.", call. = FALSE)
    }
    return(idx)
  }

  stop("Unsupported index type.", call. = FALSE)
}

#' Normalize index vector to 0-based integers for MLX
#'
#' @param idx Index vector (integer, logical, or mlx array).
#' @param dim_size Integer size of the dimension being indexed.
#' @return Integer vector of 0-based indices, or `NULL` if `idx` is `NULL`.
#' @noRd
.normalize_index_vector <- function(idx, dim_size) {
  idx_1based <- .resolve_to_positive_indices(idx, dim_size)
  if (is.null(idx_1based) || length(idx_1based) == 0L) {
    return(idx_1based)
  }
  as.integer(idx_1based - 1L)
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
