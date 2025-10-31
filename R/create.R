#' Create arrays of zeros on MLX devices
#'
#' @inheritParams mlx_creation_params
#' @param dtype MLX dtype to use. One of `"float32"`, `"float64"`, `"int8"`,
#'   `"int16"`, `"int32"`, `"int64"`, `"uint8"`, `"uint16"`, `"uint32"`,
#'   `"uint64"`, `"bool"`, or `"complex64"`.
#' @return An mlx array filled with zeros.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.zeros}
#' @export
#' @examples
#' zeros <- mlx_zeros(c(2, 3))
#' zeros_int <- mlx_zeros(c(2, 3), dtype = "int32")
mlx_zeros <- function(dim,
                      dtype = c("float32", "float64", "int8", "int16", "int32", "int64",
                               "uint8", "uint16", "uint32", "uint64", "bool", "complex64"),
                      device = mlx_default_device()) {
  dim <- .validate_shape(dim)
  dtype <- match.arg(dtype)
  handle <- .mlx_resolve_device(device, mlx_default_device())
  ptr <- .mlx_eval_with_stream(handle, function(dev) cpp_mlx_zeros(dim, dtype, dev))
  .mlx_wrap_result(ptr, handle$device)
}

#' Create arrays of ones on MLX devices
#'
#' @inheritParams mlx_zeros
#' @return An mlx array filled with ones.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.ones}
#' @export
#' @examples
#' ones <- mlx_ones(c(2, 2), dtype = "float64", device = "cpu")
#' ones_int <- mlx_ones(c(3, 3), dtype = "int32")
mlx_ones <- function(dim,
                     dtype = c("float32", "float64", "int8", "int16", "int32", "int64",
                              "uint8", "uint16", "uint32", "uint64", "bool", "complex64"),
                     device = mlx_default_device()) {
  dim <- .validate_shape(dim)
  dtype <- match.arg(dtype)
  handle <- .mlx_resolve_device(device, mlx_default_device())
  ptr <- .mlx_eval_with_stream(handle, function(dev) cpp_mlx_ones(dim, dtype, dev))
  .mlx_wrap_result(ptr, handle$device)
}

#' Zeros shaped like an existing mlx array
#'
#' `mlx_zeros_like()` mirrors [`mlx.core.zeros_like()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.zeros_like):
#' it creates a zero-filled array matching the source array's shape. Optionally override the dtype
#' or device.
#'
#' @inheritParams mlx_array_required
#' @param dtype Optional MLX dtype override. Defaults to the source array's dtype.
#' @inheritParams common_params
#' @return An mlx array of zeros matching `x`.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.zeros_like}
#' @export
#' @examples
#' base <- mlx_ones(c(2, 2))
#' zeros <- mlx_zeros_like(base)
#' as.matrix(zeros)
mlx_zeros_like <- function(x,
                           dtype = NULL,
                           device = NULL) {
  x <- as_mlx(x)
  valid_dtypes <- c(
    "float32", "float64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool", "complex64"
  )

  dtype <- if (is.null(dtype)) {
    x$dtype
  } else {
    match.arg(dtype, valid_dtypes)
  }

  target_device <- if (is.null(device)) x$device else device
  handle <- .mlx_resolve_device(target_device, x$device)
  ptr <- .mlx_eval_with_stream(handle, function(dev) cpp_mlx_zeros_like(x$ptr, dtype, dev))
  .mlx_wrap_result(ptr, handle$device)
}

#' Ones shaped like an existing mlx array
#'
#' `mlx_ones_like()` mirrors [`mlx.core.ones_like()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.ones_like),
#' creating an array of ones with the same shape. Optionally override dtype or device.
#'
#' @inheritParams mlx_array_required
#' @param dtype Optional MLX dtype override. Defaults to the source array's dtype.
#' @inheritParams common_params
#' @return An mlx array of ones matching `x`.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.ones_like}
#' @export
#' @examples
#' base <- mlx_full(c(2, 3), 5)
#' ones <- mlx_ones_like(base)
#' as.matrix(ones)
mlx_ones_like <- function(x,
                          dtype = NULL,
                          device = NULL) {
  x <- as_mlx(x)
  valid_dtypes <- c(
    "float32", "float64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool", "complex64"
  )

  dtype <- if (is.null(dtype)) {
    x$dtype
  } else {
    match.arg(dtype, valid_dtypes)
  }

  target_device <- if (is.null(device)) x$device else device
  handle <- .mlx_resolve_device(target_device, x$device)
  ptr <- .mlx_eval_with_stream(handle, function(dev) cpp_mlx_ones_like(x$ptr, dtype, dev))
  .mlx_wrap_result(ptr, handle$device)
}

#' Fill an mlx array with a constant value
#'
#' @param value Scalar value used to fill the array. Numeric, logical, or complex.
#' @inheritParams mlx_zeros
#' @param dtype MLX dtype (`"float32"`, `"float64"`, `"bool"`, or `"complex64"`).
#'   If omitted, defaults to `"complex64"` for complex scalars, `"bool"` for logical scalars,
#'   and `"float32"` otherwise.
#' @return An mlx array filled with the supplied value.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.full}
#' @export
#' @examples
#' filled <- mlx_full(c(2, 2), 3.14)
#' complex_full <- mlx_full(c(2, 2), 1+2i, dtype = "complex64")
mlx_full <- function(dim,
                     value,
                     dtype = NULL,
                     device = mlx_default_device()) {
  dim <- .validate_shape(dim)
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

  handle <- .mlx_resolve_device(device, mlx_default_device())
  ptr <- .mlx_eval_with_stream(handle, function(dev) cpp_mlx_full(dim, value, dtype, dev))
  .mlx_wrap_result(ptr, handle$device)
}

#' Identity-like matrices on MLX devices
#'
#' @param n Number of rows.
#' @param m Optional number of columns (defaults to `n`).
#' @param k Diagonal index: `0` is the main diagonal, positive values shift upward,
#'   negative values shift downward.
#' @inheritParams mlx_zeros
#' @return An mlx matrix with ones on the selected diagonal and zeros elsewhere.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.eye}
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
  handle <- .mlx_resolve_device(device, mlx_default_device())
  ptr <- .mlx_eval_with_stream(handle, function(dev) cpp_mlx_eye(n, m, k, dtype, dev))
  .mlx_wrap_result(ptr, handle$device)
}

#' Identity matrices on MLX devices
#'
#' @param n Size of the square matrix.
#' @inheritParams mlx_eye
#' @return An mlx identity matrix.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.identity}
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
  handle <- .mlx_resolve_device(device, mlx_default_device())
  ptr <- .mlx_eval_with_stream(handle, function(dev) cpp_mlx_identity(n, dtype, dev))
  .mlx_wrap_result(ptr, handle$device)
}

#' Triangular helpers for MLX arrays
#'
#' `mlx_tri()` creates a lower-triangular mask (ones on and below a diagonal,
#' zeros elsewhere). `mlx_tril()` and `mlx_triu()` retain only the lower or
#' upper triangular part of an existing array, respectively.
#'
#' @inheritParams mlx_eye
#' @param m Optional number of columns (defaults to `n` for square output).
#' @param k Diagonal offset: `0` selects the main diagonal, positive values move
#'   to the upper diagonals, negative values to the lower diagonals.
#' @param dtype MLX dtype to use (`"float32"` or `"float64"`).
#' @param x Object coercible to `mlx`.
#' @return An `mlx` array.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.tri}
#' @export
#' @examples
#' mlx_tri(3)          # 3x3 lower-triangular mask
#' mlx_tril(diag(3) + 2)  # keep lower part of a matrix
mlx_tri <- function(n,
                    m = NULL,
                    k = 0L,
                    dtype = c("float32", "float64"),
                    device = mlx_default_device()) {
  n <- as.integer(n)
  if (length(n) != 1L || is.na(n) || n <= 0L) {
    stop("n must be a positive integer.", call. = FALSE)
  }

  if (is.null(m)) {
    m_arg <- NULL
  } else {
    m_arg <- as.integer(m)
    if (length(m_arg) != 1L || is.na(m_arg) || m_arg <= 0L) {
      stop("m must be a positive integer when supplied.", call. = FALSE)
    }
  }

  k <- as.integer(k)
  if (length(k) != 1L || is.na(k)) {
    stop("k must be a single integer.", call. = FALSE)
  }

  dtype <- match.arg(dtype)
  handle <- .mlx_resolve_device(device, mlx_default_device())
  ptr <- .mlx_eval_with_stream(handle, function(dev) cpp_mlx_tri(n, m_arg, k, dtype, dev))
  .mlx_wrap_result(ptr, handle$device)
}

#' @rdname mlx_tri
#' @export
mlx_tril <- function(x, k = 0L) {
  x <- as_mlx(x)

  k <- as.integer(k)
  if (length(k) != 1L || is.na(k)) {
    stop("k must be a single integer.", call. = FALSE)
  }

  ptr <- cpp_mlx_tril(x$ptr, k, x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_tri
#' @export
mlx_triu <- function(x, k = 0L) {
  x <- as_mlx(x)

  k <- as.integer(k)
  if (length(k) != 1L || is.na(k)) {
    stop("k must be a single integer.", call. = FALSE)
  }

  ptr <- cpp_mlx_triu(x$ptr, k, x$device)
  .mlx_wrap_result(ptr, x$device)
}


#' Diagonal matrix extraction and construction
#'
#' Generic function for extracting/constructing diagonal matrices.
#' @param x An object.
#' @param nrow,ncol Optional dimensions for matrix construction.
#' @param names Logical indicating whether to use names.
#' @export
diag <- function(x = 1, nrow, ncol, names = TRUE) {
  UseMethod("diag")
}

#' @export
diag.default <- function(x, ...) base::diag(x, ...)


#' @export
#' @rdname mlx_diagonal
#' @param names Unused.
#' @param nrow,ncol Diagonal offset (nrow only; ncol ignored).
#'
#' `diag.mlx()` is an R interface to `mlx_diagonal()` with the same semantics
#' as [base::diag()].
diag.mlx <- function(x, nrow, ncol, names = TRUE) {
  x <- as_mlx(x)

  # Determine k offset if nrow is specified
  k <- 0L
  if (!missing(nrow)) {
    k <- as.integer(nrow)
  }

  ptr <- cpp_mlx_diag(x$ptr, k, x$device)
  dim <- cpp_mlx_shape(ptr)
  dtype <- cpp_mlx_dtype(ptr)
  new_mlx(ptr, dim, dtype, x$device)
}

#' Numerical ranges on MLX devices
#'
#' `mlx_arange()` mirrors `base::seq()` with mlx arrays: it creates evenly spaced values
#' starting at `start` (default `0`), stepping by `step` (default `1`), and stopping before `stop`.
#'
#' @param stop Exclusive upper bound.
#' @param start Optional starting value (defaults to 0).
#' @param step Optional step size (defaults to 1).
#' @param dtype MLX dtype (`"float32"` or `"float64"`).
#' @inheritParams mlx_zeros
#' @return A 1D mlx array.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.arange}
#' @export
#' @examples
#' mlx_arange(5)                    # 0, 1, 2, 3, 4
#' mlx_arange(5, start = 1, step = 2) # 1, 3
mlx_arange <- function(stop,
                       start = NULL,
                       step = NULL,
                       dtype = c("float32", "float64", "int8", "int16", "int32", "int64",
                                "uint8", "uint16", "uint32", "uint64"),
                       device = mlx_default_device()) {
  if (!length(stop) || length(stop) != 1L) {
    stop("stop must be a single numeric value.", call. = FALSE)
  }

  dtype <- match.arg(dtype)
  handle <- .mlx_resolve_device(device, mlx_default_device())

  start_arg <- if (is.null(start)) NULL else as.numeric(start)
  step_arg <- if (is.null(step)) NULL else as.numeric(step)

  if (!is.null(start_arg) && length(start_arg) != 1L) {
    stop("start must be NULL or a single numeric value.", call. = FALSE)
  }
  if (!is.null(step_arg) && length(step_arg) != 1L) {
    stop("step must be NULL or a single numeric value.", call. = FALSE)
  }

  ptr <- .mlx_eval_with_stream(handle, function(dev) {
    cpp_mlx_arange(start_arg, as.numeric(stop), step_arg, dtype, dev)
  })
  .mlx_wrap_result(ptr, handle$device)
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
#' @return A 1D mlx array.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.linspace}
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
  handle <- .mlx_resolve_device(device, mlx_default_device())

  ptr <- .mlx_eval_with_stream(handle, function(dev) {
    cpp_mlx_linspace(
      as.numeric(start),
      as.numeric(stop),
      as.integer(num),
      dtype,
      dev
    )
  })
  .mlx_wrap_result(ptr, handle$device)
}

# Helper to validate shapes ----------------------------------------------------

#' Validate and coerce shape specification
#'
#' @param dim Integer or numeric vector of dimension sizes.
#' @return Integer vector of positive dimensions.
#' @noRd
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
