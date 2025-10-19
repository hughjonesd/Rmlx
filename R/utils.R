#' Print MLX array
#'
#' @param x An \code{mlx} object
#' @param ... Additional arguments (ignored)
#' @export
#' @method print mlx
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
#' @param object An \code{mlx} object
#' @param ... Additional arguments (ignored)
#' @export
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
#' @param x An \code{mlx} object
#' @return Integer vector of dimensions
#' @export
#' @method dim mlx
dim.mlx <- function(x) {
  x$dim
}

#' Get length of MLX array
#'
#' @param x An \code{mlx} object
#' @return Total number of elements
#' @export
#' @method length mlx
length.mlx <- function(x) {
  prod(x$dim)
}

#' Get dimensions helper
#'
#' @param x An \code{mlx} object
#' @return Dimensions
#' @export
mlx_dim <- function(x) {
  stopifnot(is.mlx(x))
  x$dim
}

#' Get data type helper
#'
#' @param x An \code{mlx} object
#' @return Data type string
#' @export
mlx_dtype <- function(x) {
  stopifnot(is.mlx(x))
  x$dtype
}

#' Subset MLX array
#'
#' @param x An \code{mlx} object
#' @param i Row indices
#' @param j Column indices (for matrices)
#' @param ... Additional indices
#' @param drop Should dimensions be dropped? (default: TRUE)
#' @return Subsetted \code{mlx} object
#' @export
#' @method [ mlx
`[.mlx` <- function(x, i, j, ..., drop = TRUE) {
  ndim <- length(x$dim)

  # Handle 1D case
  if (ndim == 1) {
    indices <- .normalize_index(i, x$dim[1])
    starts <- indices$start
    stops <- indices$stop
    strides <- indices$stride

    ptr <- cpp_mlx_slice(x$ptr, starts, stops, strides)
    new_dim <- max(0, (stops[1] - starts[1]) %/% strides[1])

    return(new_mlx(ptr, new_dim, x$dtype, x$device))
  }

  # Handle 2D case
  if (ndim == 2) {
    # Default to all rows/cols if not specified
    if (missing(i)) i <- seq_len(x$dim[1])
    if (missing(j)) j <- seq_len(x$dim[2])

    idx_i <- .normalize_index(i, x$dim[1])
    idx_j <- .normalize_index(j, x$dim[2])

    starts <- c(idx_i$start, idx_j$start)
    stops <- c(idx_i$stop, idx_j$stop)
    strides <- c(idx_i$stride, idx_j$stride)

    ptr <- cpp_mlx_slice(x$ptr, starts, stops, strides)

    new_dim <- c(
      max(0, (stops[1] - starts[1]) %/% strides[1]),
      max(0, (stops[2] - starts[2]) %/% strides[2])
    )

    if (drop) {
      new_dim <- new_dim[new_dim > 1]
      if (length(new_dim) == 0) new_dim <- 1L
    }

    return(new_mlx(ptr, new_dim, x$dtype, x$device))
  }

  stop("Indexing for arrays with >2 dimensions not yet implemented")
}

# Internal helper: normalize index to start/stop/stride
.normalize_index <- function(idx, dim_size) {
  if (missing(idx) || is.null(idx)) {
    # Full span
    return(list(start = 0L, stop = as.integer(dim_size), stride = 1L))
  }

  if (is.logical(idx)) {
    idx <- which(idx)
  }

  if (is.numeric(idx)) {
    idx <- as.integer(idx)

    # Handle negative indices
    if (any(idx < 0)) {
      stop("Negative indices not yet supported")
    }

    # Convert to 0-indexed
    idx <- idx - 1L

    # For now, assume contiguous range
    if (length(idx) == 0) {
      return(list(start = 0L, stop = 0L, stride = 1L))
    }

    start <- min(idx)
    stop <- max(idx) + 1L
    stride <- if (length(idx) > 1) {
      unique_diffs <- unique(diff(idx))
      if (length(unique_diffs) == 1) unique_diffs else 1L
    } else {
      1L
    }

    return(list(
      start = as.integer(start),
      stop = as.integer(stop),
      stride = as.integer(stride)
    ))
  }

  stop("Unsupported index type")
}
