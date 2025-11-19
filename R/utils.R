
#' Wrap a raw MLX pointer into an mlx object
#'
#' @param ptr External pointer returned by C++ bindings.
#' @param device Device string associated with the array.
#' @return An mlx array.
#' @noRd
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


#' Print MLX array
#'
#' Printing an array only evaluates it if it is of small size (less than
#' 100 elements and 2 dimensions)
#'
#' @inheritParams common_params
#' @param ... Additional arguments (ignored)
#' @return `x`, invisibly.
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' print(x)
print.mlx <- function(x, ...) {
  cat(sprintf("mlx array [%s]\n", paste(mlx_shape(x), collapse = " x ")))
  cat(sprintf("  dtype: %s\n", mlx_dtype(x)))
  cat(sprintf("  device: %s\n", x$device))

  # Show preview for small arrays
  total_size <- length(x)
  if (total_size <= 100 && length(mlx_shape(x)) <= 2) {
    cat("  values:\n")
    mat <- as.array(x)
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
#' @return `NULL` invisibly.
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
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
#' x <- as_mlx(matrix(1:4, 2, 2))
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
  if (!is.numeric(value) || any(is.na(value))) {
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

  if (!is.numeric(newshape) || any(is.na(newshape))) {
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
#' x <- as_mlx(matrix(1:6, 2, 3))
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
#' x <- as_mlx(matrix(1:6, 2, 3))
#' mlx_dtype(x)
mlx_dtype <- function(x) {
  stopifnot(is_mlx(x))
  cpp_mlx_dtype(x$ptr)
}
