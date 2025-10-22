#' Create tensors of zeros on MLX devices
#'
#' @inheritParams mlx_params
#' @return An `mlx` tensor filled with zeros.
#' @export
#' @examples
#' zeros <- mlx_zeros(c(2, 3))
mlx_zeros <- function(dim,
                      dtype = c("float32", "float64"),
                      device = mlx_default_device()) {
  dim <- .validate_shape(dim)
  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_zeros(dim, dtype, device)
  .mlx_wrap_result(ptr, device)
}

#' Create tensors of ones on MLX devices
#'
#' @inheritParams mlx_zeros
#' @return An `mlx` tensor filled with ones.
#' @export
#' @examples
#' ones <- mlx_ones(c(2, 2), dtype = "float64", device = "cpu")
mlx_ones <- function(dim,
                     dtype = c("float32", "float64"),
                     device = mlx_default_device()) {
  dim <- .validate_shape(dim)
  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_ones(dim, dtype, device)
  .mlx_wrap_result(ptr, device)
}

#' Fill an MLX tensor with a constant value
#'
#' @param value Scalar value used to fill the tensor. Numeric, logical, or complex.
#' @inheritParams mlx_zeros
#' @param dtype MLX dtype (`"float32"`, `"float64"`, `"bool"`, or `"complex64"`).
#'   If omitted, defaults to `"complex64"` for complex scalars, `"bool"` for logical scalars,
#'   and `"float32"` otherwise.
#' @return An `mlx` tensor filled with the supplied value.
#' @export
#' @examples
#' filled <- mlx_full(c(2, 2), 3.14)
#' complex_full <- mlx_full(c(2, 2), 1+2i, dtype = "complex64")
mlx_full <- function(dim,
                     value,
                     dtype = NULL,
                     device = mlx_default_device()) {
  dim <- .validate_shape(dim)
  device <- match.arg(device, c("gpu", "cpu"))
  if (length(value) != 1) {
    stop("value must be a scalar.", call. = FALSE)
  }

  valid_dtypes <- c("float32", "float64", "complex64", "bool")

  if (is.null(dtype)) {
    dtype <- if (is.complex(value)) {
      "complex64"
    } else if (is.logical(value)) {
      "bool"
    } else {
      "float32"
    }
  } else {
    dtype <- match.arg(dtype, valid_dtypes)
  }

  if (!(dtype %in% valid_dtypes)) {
    stop("Unsupported dtype: ", dtype, call. = FALSE)
  }

  ptr <- cpp_mlx_full(dim, value, dtype, device)
  .mlx_wrap_result(ptr, device)
}

#' Identity-like matrices on MLX devices
#'
#' @param n Number of rows.
#' @param m Optional number of columns (defaults to `n`).
#' @param k Diagonal index: `0` is the main diagonal, positive values shift upward,
#'   negative values shift downward.
#' @inheritParams mlx_zeros
#' @return An `mlx` matrix with ones on the selected diagonal and zeros elsewhere.
#' @export
#' @examples
#' eye <- mlx_eye(3)
#' upper_eye <- mlx_eye(3, k = 1)
mlx_eye <- function(n,
                    m = n,
                    k = 0L,
                    dtype = c("float32", "float64"),
                    device = mlx_default_device()) {
  n <- as.integer(n)
  m <- as.integer(m)
  k <- as.integer(k)

  if (length(n) != 1L || n <= 0) {
    stop("n must be a positive integer.", call. = FALSE)
  }
  if (length(m) != 1L || m <= 0) {
    stop("m must be a positive integer.", call. = FALSE)
  }

  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_eye(n, m, k, dtype, device)
  .mlx_wrap_result(ptr, device)
}

#' Identity matrices on MLX devices
#'
#' @param n Size of the square matrix.
#' @inheritParams mlx_eye
#' @return An `mlx` identity matrix.
#' @export
#' @examples
#' I4 <- mlx_identity(4)
mlx_identity <- function(n,
                         dtype = c("float32", "float64"),
                         device = mlx_default_device()) {
  n <- as.integer(n)
  if (length(n) != 1L || n <= 0) {
    stop("n must be a positive integer.", call. = FALSE)
  }

  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_identity(n, dtype, device)
  .mlx_wrap_result(ptr, device)
}

#' Numerical ranges on MLX devices
#'
#' `mlx_arange()` mirrors `base::seq()` with MLX tensors: it creates evenly spaced values
#' starting at `start` (default `0`), stepping by `step` (default `1`), and stopping before `stop`.
#'
#' @param stop Exclusive upper bound.
#' @param start Optional starting value (defaults to 0).
#' @param step Optional step size (defaults to 1).
#' @param dtype MLX dtype (`"float32"` or `"float64"`).
#' @inheritParams mlx_zeros
#' @return A 1D `mlx` tensor.
#' @export
#' @examples
#' mlx_arange(5)                    # 0, 1, 2, 3, 4
#' mlx_arange(5, start = 1, step = 2) # 1, 3
mlx_arange <- function(stop,
                       start = NULL,
                       step = NULL,
                       dtype = c("float32", "float64"),
                       device = mlx_default_device()) {
  if (!length(stop) || length(stop) != 1L) {
    stop("stop must be a single numeric value.", call. = FALSE)
  }

  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  start_arg <- if (is.null(start)) NULL else as.numeric(start)
  step_arg <- if (is.null(step)) NULL else as.numeric(step)

  if (!is.null(start_arg) && length(start_arg) != 1L) {
    stop("start must be NULL or a single numeric value.", call. = FALSE)
  }
  if (!is.null(step_arg) && length(step_arg) != 1L) {
    stop("step must be NULL or a single numeric value.", call. = FALSE)
  }

  ptr <- cpp_mlx_arange(start_arg, as.numeric(stop), step_arg, dtype, device)
  .mlx_wrap_result(ptr, device)
}

#' Evenly spaced ranges on MLX devices
#'
#' `mlx_linspace()` creates `num` evenly spaced values from `start` to `stop`, inclusive.
#' Unlike `mlx_arange()`, you specify how many samples you want rather than the step size.
#'
#' @param start Starting value.
#' @param stop Final value (inclusive).
#' @param num Number of samples to generate.
#' @inheritParams mlx_arange
#' @return A 1D `mlx` tensor.
#' @export
#' @examples
#' mlx_linspace(0, 1, num = 5)
mlx_linspace <- function(start,
                         stop,
                         num = 50L,
                         dtype = c("float32", "float64"),
                         device = mlx_default_device()) {
  if (length(num) != 1L || num <= 0) {
    stop("num must be a positive integer.", call. = FALSE)
  }

  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_linspace(
    as.numeric(start),
    as.numeric(stop),
    as.integer(num),
    dtype,
    device
  )
  .mlx_wrap_result(ptr, device)
}

# Helper to validate shapes ----------------------------------------------------
.validate_shape <- function(dim) {
  if (length(dim) == 0L) {
    stop("dim must contain at least one element.", call. = FALSE)
  }
  dim <- as.integer(dim)
  if (any(is.na(dim)) || any(dim <= 0)) {
    stop("dim must contain positive integers.", call. = FALSE)
  }
  dim
}
