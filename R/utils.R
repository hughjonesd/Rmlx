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
#' @param i Row indices
#' @param j Column indices (for matrices)
#' @param ... Additional indices
#' @param drop Should dimensions be dropped? (default: TRUE)
#' @return Subsetted `mlx` object
#' @export
#' @method [ mlx
#' @examples
#' x <- as_mlx(matrix(1:9, 3, 3))
#' x[1, ]
`[.mlx` <- function(x, i, j, ..., drop = TRUE) {
  ndim <- length(x$dim)

  # Handle 1D case
  if (ndim == 1) {
    indices <- .normalize_index(i, x$dim[1])
    starts <- indices$start
    stops <- indices$stop
    strides <- indices$stride

    ptr <- cpp_mlx_slice(x$ptr, starts, stops, strides)
    new_len <- .slice_extent(starts[1], stops[1], strides[1])

    return(new_mlx(ptr, as.integer(new_len), x$dtype, x$device))
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
      .slice_extent(starts[1], stops[1], strides[1]),
      .slice_extent(starts[2], stops[2], strides[2])
    )

    if (drop) {
      keep <- new_dim != 1L
      new_dim <- new_dim[keep]
      if (length(new_dim) == 0) new_dim <- 1L
    }

    return(new_mlx(ptr, as.integer(new_dim), x$dtype, x$device))
  }

  stop("Indexing for arrays with >2 dimensions not yet implemented")
}

#' Normalize indexing parameters for slicing
#'
#' @param idx Index specification from R.
#' @param dim_size Size of the dimension.
#' @return List with `start`, `stop`, `stride` (zero-based).
#' @noRd
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

#' Compute the extent of a slice given start/stop/stride
#'
#' @param start,stop,stride Slice parameters.
#' @return Length of the resulting slice.
#' @noRd
.slice_extent <- function(start, stop, stride) {
  if (stride <= 0L) {
    stop("Stride must be positive")
  }
  delta <- stop - start
  if (delta <= 0L) {
    return(0L)
  }
  as.integer(((delta - 1L) %/% stride) + 1L)
}
