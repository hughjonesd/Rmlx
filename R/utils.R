#' Common MLX parameter documentation
#'
#' @param x An `mlx` tensor, matrix, array, or object coercible to `mlx`.
#' @param dim Integer vector giving the tensor shape.
#' @param dtype Desired MLX dtype ("float32" or "float64").
#' @param device Target device ("gpu" or "cpu").
#' @name mlx_params
NULL

#' Print MLX array
#'
#' @param x An `mlx` object
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
#' @param object An `mlx` object
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
#' @param x An `mlx` object
#' @return Integer vector of dimensions
#' @export
#' @method dim mlx
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' dim(x)
dim.mlx <- function(x) {
  x$dim
}

#' Get length of MLX array
#'
#' @param x An `mlx` object
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
#' @param x An `mlx` object
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
#' @param x An `mlx` object
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
#' @param x An `mlx` object
#' @param ... Indices for each dimension. Provide one per axis; omitted indices
#'   select the full extent. Logical indices recycle to the dimension length.
#' @param drop Should dimensions be dropped? (default: FALSE)
#' @return Subsetted `mlx` object
#' @export
#' @method [ mlx
#' @examples
#' x <- as_mlx(matrix(1:9, 3, 3))
#' x[1, ]
`[.mlx` <- function(x, ..., drop = FALSE) {
  ndim <- length(x$dim)
  if (ndim == 0L) {
    stop("Cannot subset a scalar mlx tensor.", call. = FALSE)
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

  out <- x
  for (axis in seq_len(ndim)) {
    idx <- if (axis <= length(idx_list)) idx_list[[axis]] else NULL
    sel <- .normalize_index_vector(idx, out$dim[axis])
    if (is.null(sel)) next

    ptr <- cpp_mlx_take(out$ptr, sel, axis - 1L)
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

.normalize_index_vector <- function(idx, dim_size) {
  if (is.null(idx)) {
    return(NULL)
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
    idx <- as.integer(idx)
    if (any(idx < 0L)) {
      if (any(idx > 0L)) {
        stop("Cannot mix positive and negative indices.", call. = FALSE)
      }
      keep <- setdiff(seq_len(dim_size), abs(idx))
      idx <- keep
    }
    if (any(idx < 1L) || any(idx > dim_size)) {
      stop("Index out of bounds.", call. = FALSE)
    }
    return(as.integer(idx - 1L))
  }

  stop("Unsupported index type.", call. = FALSE)
}
