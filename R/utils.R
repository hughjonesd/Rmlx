
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
  cat(sprintf("mlx array [%s]\n", paste(dim(x), collapse = " x ")))
  cat(sprintf("  dtype: %s\n", x$dtype))
  cat(sprintf("  device: %s\n", x$device))

  # Show preview for small arrays
  total_size <- prod(dim(x))
  if (total_size <= 100 && length(dim(x)) <= 2) {
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
    paste(dim(object), collapse = " x "),
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
  cpp_mlx_shape(x$ptr)
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

  # Special case: setting dim to integer(0) means convert to 1D vector
  if (length(value) == 0) {
    current_size <- prod(dim(x))
    return(mlx_reshape(x, current_size))
  }

  # Check that product matches
  current_size <- prod(dim(x))
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

  current_size <- prod(dim(x))
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
  prod(dim(x))
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
  dim(x)
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
