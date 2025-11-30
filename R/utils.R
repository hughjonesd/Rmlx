
#' Wrap a raw MLX pointer into an mlx object
#'
#' @param ptr External pointer returned by C++ bindings.
#' @param device Device string associated with the array.
#' @return An mlx array.
#' @noRd
is_mlx_stream <- function(x) inherits(x, "mlx_stream")

resolve_device <- function(device, default = mlx_default_device()) {
  if (missing(device) || is.null(device)) {
    device <- default
  }

  if (is_mlx_stream(device)) {
    return(list(device = device$device, stream_ptr = device$ptr))
  }

  if (!is.character(device) || length(device) != 1L) {
    stop('device must be "gpu", "cpu", or an mlx_stream.', call. = FALSE)
  }

  device_chr <- match.arg(device, c("gpu", "cpu"))
  list(device = device_chr, stream_ptr = NULL)
}

eval_with_stream <- function(handle, fn) {
  if (is.null(handle$stream_ptr)) {
    return(fn(handle$device))
  }

  old <- cpp_mlx_stream_default(handle$device)
  on.exit(cpp_mlx_set_default_stream(old), add = TRUE)
  cpp_mlx_set_default_stream(handle$stream_ptr)
  fn(handle$device)
}

#' Check for duplicated rows in an mlx matrix
#'
#' NB: this will run slow, or out of memory, for more than about 1e4 rows.
#' @param x a mlx matrix (only!)
#' @returns TRUE or FALSE
#' @noRd
duplicated_rows <- function(x) {
  shape <- mlx_shape(x)
  x_rows <- mlx_reshape(x, c(1L, shape))
  x_cols <- mlx_reshape(x, c(shape[1L], 1L, shape[2L]))
  # comp[i, j, k] says whether row i == row j on column k
  comp <- x_rows == x_cols
  # collapse the columns: is i == j on all columns?
  eq <- mlx_all(comp, axes = 3)
  # remove diagonal and below-diagonal rows
  upper <- mlx_triu(eq, k = 1L)
  any(upper) # if you wanted a list of duplicate rows, add axis = 1
}

#' Check for duplicated rows
#'
#' @param x An mlx matrix
#' @returns TRUE/FALSE
#' @noRd
duplicated_rows_lex <- function(x) {
  shape <- mlx_shape(x)
  stopifnot(length(shape) == 2L)
  n  <- shape[1]

  # 2. Apply that order to X to get lex-sorted rows
  x_sorted <- lex_sort(x)

  # 3. Compare each row in sorted order to its predecessor
  # same_as_prev[i] = TRUE iff XS[i, ] == XS[i-1, ]
  same_as_prev <- mlx_zeros(n, dtype = "bool")
  if (n > 1L) {
    same_as_prev[2:n] <- mlx_all(
      x_sorted[2:n, , drop = FALSE] == x_sorted[1:(n - 1L), , drop = FALSE],
      axes = 2L
    )
  }
  any(same_as_prev)
  # If we wanted to find the duplicates:
  # 4. Map back from sorted order to original row order
  # dup <- mlx_zeros(n, dtype = "bool")
  # dup[order] <- same_as_prev
  # dup
}

#' Lexicographically sort rows of x
#'
#' @param x An mlx matrix (only!)
#' @returns row-sorted x
#' @noRd
lex_sort <- function (x) {
  shape <- mlx_shape(x)
  stopifnot(length(shape) == 2L)
  n  <- shape[1]
  d  <- shape[2]
  order <- mlx_arange(1L, n)

  # Loop columns from last to first:
  # refine `order` using argsort on each column.
  for (j in seq.int(d, 1L)) {
    col  <- x[order, j, drop = FALSE]    # column j in the current order
    idx  <- mlx_argsort(col)             # permutation of 0..(n-1) for this column
    order <- order[idx]                  # refine the lexicographic order
  }

  # 2. Apply that order to X to get lex-sorted rows
  x[order, , drop = FALSE]
}

#' Print MLX array
#'
#' Printing an array only evaluates it if it is of small size (less than
#' 100 elements and 2 dimensions)
#'
#' @inheritParams common_params
#' @inheritParams ellipsis_ignored
#' @return `x`, invisibly.
#' @export
#' @examples
#' x <- mlx_matrix(1:4, 2, 2)
#' print(x)
print.mlx <- function(x, ...) {
  cat(sprintf("mlx array [%s]\n", paste(mlx_shape(x), collapse = " x ")))
  cat(sprintf("  dtype: %s\n", mlx_dtype(x)))
  cat(sprintf("  device: %s\n", x$device))

  # Show preview for small arrays
  total_size <- length(x)
  if (total_size <= 100 && length(mlx_shape(x)) <= 2) {
    cat("  values:\n")
    mat <- as_r(x)
    print(mat)
  } else {
    cat(sprintf("  (%d elements, not shown)\n", total_size))
  }

  invisible(x)
}

#' Object structure for MLX array
#'
#' @param object An mlx object
#' @inheritParams ellipsis_ignored
#' @return `NULL` invisibly.
#' @export
#' @examples
#' x <- mlx_matrix(1:4, 2, 2)
#' str(x)
str.mlx <- function(object, ...) {
  cat(sprintf(
    "mlx [%s] %s on %s\n",
    paste(mlx_shape(object), collapse = " x "),
    mlx_dtype(object),
    object$device
  ))
  invisible(NULL)
}

#' Get dimensions of MLX array
#'
#' `dim()` mirrors base R semantics and returns `NULL` for 1-D vectors and
#' scalars, while [`mlx_shape()`] always returns the raw MLX shape (integers,
#' never `NULL`). Use `mlx_shape()` when you need the underlying MLX dimension
#' metadata and `dim()` when you want R-like behaviour.
#'
#' @inheritParams common_params
#' @return For `dim()`, an integer vector of dimensions or `NULL` for vectors/
#'   scalars. For `mlx_shape()`, an integer vector (length zero for scalars).
#' @export
#' @examples
#' x <- mlx_matrix(1:4, 2, 2)
#' dim(x)
#'
#' v <- as_mlx(1:3)
#' dim(v)        # NULL (matches base R)
#' mlx_shape(v)  # 3
dim.mlx <- function(x) {
  shape <- cpp_mlx_shape(x$ptr)
  if (length(shape) <= 1L) {
    return(NULL)
  }
  shape
}

#' @rdname dim.mlx
#' @export
mlx_shape <- function(x) {
  stopifnot(is_mlx(x))
  cpp_mlx_shape(x$ptr)
}

#' Set dimensions of MLX array
#'
#' Reshapes the MLX array to the specified dimensions. The total number of
#' elements must remain the same.
#'
#' @inheritParams common_params
#' @param value Integer vector of new dimensions
#' @return Reshaped mlx object.
#' @export
#' @seealso [mlx_reshape()]
#' @examples
#' x <- as_mlx(1:12)
#' dim(x) <- c(3, 4)
#' dim(x)
`dim<-.mlx` <- function(x, value) {
  if (!is.numeric(value) || anyNA(value)) {
    stop("dims must be a numeric vector without NAs", call. = FALSE)
  }

  value <- as.integer(value)

  if (any(value < 0)) {
    stop("dims cannot be negative", call. = FALSE)
  }

  # Special case: setting dim to integer(0) means convert to 1D vector
  if (length(value) == 0) {
    current_size <- length(x)
    return(mlx_reshape(x, current_size))
  }

  # Check that product matches
  current_size <- length(x)
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

  if (!is.numeric(newshape) || anyNA(newshape)) {
    stop("newshape must be a numeric vector without NAs", call. = FALSE)
  }

  newshape <- as.integer(newshape)

  if (any(newshape < 0)) {
    stop("newshape cannot contain negative values", call. = FALSE)
  }

  current_size <- length(x)
  new_size <- prod(newshape)

  if (current_size != new_size) {
    stop(sprintf(
      "Cannot reshape array of size %d into shape with size %d",
      current_size, new_size
    ), call. = FALSE)
  }

  ptr <- cpp_mlx_reshape(x$ptr, newshape)
  new_mlx(ptr, x$device)
}

#' Get length of MLX array
#'
#' @inheritParams common_params
#' @return Total number of elements.
#' @export
#' @examples
#' x <- mlx_matrix(1:6, 2, 3)
#' length(x)
length.mlx <- function(x) {
  shape <- mlx_shape(x)
  if (length(shape) == 0L) {
    return(1L)
  }
  prod(shape)
}

#' Get the data type of an MLX array
#'
#' @inheritParams common_params
#' @return A data type string (see [as_mlx()] for possibilities).
#' @export
#' @examples
#' x <- mlx_matrix(1:6, 2, 3)
#' mlx_dtype(x)
mlx_dtype <- function(x) {
  stopifnot(is_mlx(x))
  cpp_mlx_dtype(x$ptr)
}
