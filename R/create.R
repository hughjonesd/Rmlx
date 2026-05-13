#' Create arrays of zeros on MLX devices
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @inheritParams common_params
#' @return An mlx array filled with zeros.
#' @seealso [mlx.core.zeros](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.zeros)
#' @export
#' @examples
#' zeros <- mlx_zeros(c(2, 3))
#' zeros_int <- mlx_zeros(c(2, 3), dtype = "int32")
mlx_zeros <- function(dim,
                      dtype = c("float32", "float64", "int8", "int16", "int32", "int64",
                               "uint8", "uint16", "uint32", "uint64", "bool", "complex64")) {
  dim <- validate_shape(dim)
  dtype <- match.arg(dtype)
  local_float64_cpu(dtype)
  ptr <- cpp_mlx_zeros(dim, dtype)
  new_mlx(ptr)
}

#' Create arrays of ones on MLX devices
#'
#' @inheritParams mlx_zeros
#' @return An mlx array filled with ones.
#' @seealso [mlx.core.ones](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.ones)
#' @export
#' @examples
#' ones <- with_device("cpu", mlx_ones(c(2, 2), dtype = "float64"))
#' ones_int <- mlx_ones(c(3, 3), dtype = "int32")
mlx_ones <- function(dim,
                     dtype = c("float32", "float64", "int8", "int16", "int32", "int64",
                              "uint8", "uint16", "uint32", "uint64", "bool", "complex64")) {
  dim <- validate_shape(dim)
  dtype <- match.arg(dtype)
  local_float64_cpu(dtype)
  ptr <- cpp_mlx_ones(dim, dtype)
  new_mlx(ptr)
}

#' Zeros shaped like an existing mlx array
#'
#' `mlx_zeros_like()` mirrors [`mlx.core.zeros_like()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.zeros_like):
#' it creates a zero-filled array matching the source array's shape. Optionally override the dtype
#' or dtype.
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @inheritParams mlx_array_required
#' @inheritParams common_params
#' @return An mlx array of zeros matching `x`.
#' @seealso [mlx.core.zeros_like](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.zeros_like)
#' @export
#' @examples
#' base <- mlx_ones(c(2, 2))
#' mlx_zeros_like(base)
mlx_zeros_like <- function(x,
                           dtype = NULL) {
  x <- as_mlx(x)
  valid_dtypes <- c(
    "float32", "float64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool", "complex64"
  )

  dtype <- if (is.null(dtype)) {
    mlx_dtype(x)
  } else {
    match.arg(dtype, valid_dtypes)
  }

  local_float64_cpu(dtype)
  ptr <- cpp_mlx_zeros_like(x$ptr, dtype)
  new_mlx(ptr)
}

#' Ones shaped like an existing mlx array
#'
#' `mlx_ones_like()` mirrors [`mlx.core.ones_like()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.ones_like),
#' creating an array of ones with the same shape. Optionally override dtype.
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @inheritParams mlx_array_required
#' @inheritParams common_params
#' @return An mlx array of ones matching `x`.
#' @seealso [mlx.core.ones_like](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.ones_like)
#' @export
#' @examples
#' base <- mlx_full(c(2, 3), 5)
#' mlx_ones_like(base)
mlx_ones_like <- function(x,
                          dtype = NULL) {
  x <- as_mlx(x)
  valid_dtypes <- c(
    "float32", "float64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool", "complex64"
  )

  dtype <- if (is.null(dtype)) {
    mlx_dtype(x)
  } else {
    match.arg(dtype, valid_dtypes)
  }

  local_float64_cpu(dtype)
  ptr <- cpp_mlx_ones_like(x$ptr, dtype)
  new_mlx(ptr)
}

#' Fill an mlx array with a constant value
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @param value Scalar value used to fill the array. Numeric, logical, or complex.
#' @inheritParams mlx_zeros
#' @return An mlx array filled with the supplied value.
#' @seealso [mlx.core.full](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.full)
#' @export
#' @examples
#' filled <- mlx_full(c(2, 2), 3.14)
#' complex_full <- mlx_full(c(2, 2), 1+2i, dtype = "complex64")
mlx_full <- function(dim,
                     value,
                     dtype = NULL) {
  dim <- validate_shape(dim)
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

  local_float64_cpu(dtype)
  ptr <- cpp_mlx_full(dim, value, dtype)
  new_mlx(ptr)
}

#' Identity-like matrices on MLX devices
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @param n Number of rows.
#' @param m Optional number of columns (defaults to `n`).
#' @param k Diagonal index: `0` is the main diagonal, positive values shift upward,
#'   negative values shift downward.
#' @inheritParams mlx_zeros
#' @return An mlx matrix with ones on the selected diagonal and zeros elsewhere.
#' @seealso [mlx.core.eye](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.eye)
#' @export
#' @examples
#' mlx_eye(3)
#' mlx_eye(3, k = 1)
mlx_eye <- function(n,
                    m = n,
                    k = 0L,
                    dtype = c("float32", "float64")) {
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
  local_float64_cpu(dtype)
  ptr <- cpp_mlx_eye(n, m, k, dtype)
  new_mlx(ptr)
}

#' Construct an MLX array from R data
#'
#' `mlx_array()` is a low-level constructor that skips `as_mlx()`'s type inference
#' and dimension guessing. Supply the raw payload vector plus an explicit shape
#' and it pipes the data straight into MLX.
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @inheritParams common_params
#' @param data Numeric, logical, or complex vector. `data` is recycled to
#'   match dimensions according to R rules (but with an error if it doesn't
#'   tile into the dimensions exactly).
#' @param dim Integer vector of array dimensions. Set `dim = integer(0)` for
#'   a scalar, in which case `data` must be length 1.
#' @return An `mlx` array with the requested shape.
#' @export
#' @examples
#' payload <- runif(6)
#'mlx_array(payload, dim = c(2, 3))
mlx_array <- function(data,
                      dim,
                      dtype = NULL) {
  if (!is.atomic(data) || is.list(data)) {
    stop("data must be an atomic vector.", call. = FALSE)
  }

  dim <- validate_shape(dim)
  total <- prod(dim)
  data_vec <- as.vector(data)

  data_len <- length(data_vec)
  if (data_len == 0L) {
    stop("length(data) must be positive.", call. = FALSE)
  }
  if (length(dim) == 0L) { # scalar
    if (data_len != 1L) stop("data must be length 1 for an MLX scalar.")
  } else if (data_len > total || total %% data_len != 0L) {
    stop(
      "length(data) (", data_len, ") must divide prod(dim) (", total, ").",
      call. = FALSE
    )
  }

  if (data_len < total) {
    data_vec <- rep_len(data_vec, total)
  }


  dtype_val <- if (is.null(dtype)) {
    if (is.logical(data_vec)) {
      "bool"
    } else if (is.complex(data_vec)) {
      "complex64"
    } else {
      "float32"
    }
  } else {
    match.arg(dtype,
              c(
                "float32", "float64", "bool", "complex64",
                "int8", "int16", "int32", "int64",
                "uint8", "uint16", "uint32", "uint64"
              ))
  }

  payload <- coerce_payload(data_vec, dtype_val)
  local_float64_cpu(dtype_val)
  ptr <- cpp_mlx_from_r(payload, as.integer(dim), dtype_val)
  new_mlx(ptr)
}

#' Construct MLX matrices efficiently
#'
#' `mlx_matrix()` wraps [mlx_array()] for the common 2-D case. It accepts the same
#' style arguments as [base::matrix()] but without recycling, so mistakes surface early.
#' Supply `nrow` or `ncol` (the other may be inferred from `length(data)`).
#'
#' @inheritParams mlx_array
#' @param nrow,ncol Matrix dimensions (positive integers).
#' @param byrow Logical; if `TRUE`, fill by rows (same semantics as [base::matrix()]).
#' @return An `mlx` matrix with `dim = c(nrow, ncol)`.
#' @export
#' @examples
#' mlx_matrix(1:6, nrow = 2, ncol = 3, byrow = TRUE)
mlx_matrix <- function(data,
                       nrow = NULL,
                       ncol = NULL,
                       byrow = FALSE,
                       dtype = NULL) {
  data_vec <- as.vector(data)
  total <- length(data_vec)

  if (is.null(nrow) && is.null(ncol)) {
    stop("Provide either nrow or ncol when calling mlx_matrix().", call. = FALSE)
  }

  as_dim <- function(value, name) {
    value <- as.integer(value)
    if (length(value) != 1L || is.na(value) || value <= 0) {
      stop(name, " must be a positive integer.", call. = FALSE)
    }
    value
  }

  if (is.null(nrow)) {
    ncol <- as_dim(ncol, "ncol")
    if (total %% ncol != 0) {
      stop("length(data) must be divisible by ncol.", call. = FALSE)
    }
    nrow <- total %/% ncol
  } else if (is.null(ncol)) {
    nrow <- as_dim(nrow, "nrow")
    if (total %% nrow != 0) {
      stop("length(data) must be divisible by nrow.", call. = FALSE)
    }
    ncol <- total %/% nrow
  }

  if (isTRUE(byrow)) {
    data_vec <- as.vector(matrix(data_vec, nrow = nrow, ncol = ncol, byrow = TRUE))
  }

  mlx_array(data_vec, c(nrow, ncol), dtype = dtype)
}

#' Construct MLX vectors
#'
#' `mlx_vector()` is a convenience around [mlx_array()] for 1-D payloads.
#'
#' @inheritParams mlx_array
#' @param data Atomic vector providing the elements (recycling is not allowed).
#' @return An `mlx` vector with `dim = length(data)`.
#' @export
mlx_vector <- function(data,
                       dtype = NULL) {
  if (!is.atomic(data) || is.list(data)) {
    stop("data must be an atomic vector.", call. = FALSE)
  }

  data_vec <- as.vector(data)
  if (!length(data_vec)) {
    stop("data must contain at least one element.", call. = FALSE)
  }

  mlx_array(data_vec, length(data_vec), dtype = dtype)
}

#' Construct MLX scalars
#'
#' @inheritParams mlx_array
#' @param value Single value (numeric, logical, or complex).
#' @return A dimensionless `mlx` scalar.
#' @export
mlx_scalar <- function(value,
                       dtype = NULL) {
  if (length(value) != 1L) {
    stop("value must be length 1.", call. = FALSE)
  }

  mlx_array(
    value,
    dim = integer(0),
    dtype = dtype
  )
}

#' Identity matrices on MLX devices
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @param n Size of the square matrix.
#' @inheritParams mlx_eye
#' @return An mlx identity matrix.
#' @seealso [mlx.core.identity](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.identity)
#' @export
#' @examples
#' I4 <- mlx_identity(4)
mlx_identity <- function(n,
                         dtype = c("float32", "float64")) {
  n <- as.integer(n)
  if (length(n) != 1L || n <= 0) {
    stop("n must be a positive integer.", call. = FALSE)
  }

  dtype <- match.arg(dtype)
  local_float64_cpu(dtype)
  ptr <- cpp_mlx_identity(n, dtype)
  new_mlx(ptr)
}

#' Triangular helpers for MLX arrays
#'
#' `mlx_tri()` creates a lower-triangular mask (ones on and below a diagonal,
#' zeros elsewhere). `mlx_tril()` and `mlx_triu()` retain only the lower or
#' upper triangular part of an existing array, respectively.
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @inheritParams mlx_eye
#' @param m Optional number of columns (defaults to `n` for square output).
#' @param k Diagonal offset: `0` selects the main diagonal, positive values move
#'   to the upper diagonals, negative values to the lower diagonals.
#' @param x Object coercible to `mlx`.
#' @return An `mlx` array.
#' @seealso [mlx.core.tri](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.tri)
#' @export
#' @examples
#' mlx_tri(3)          # 3x3 lower-triangular mask
#' mlx_tril(diag(3) + 2)  # keep lower part of a matrix
mlx_tri <- function(n,
                    m = NULL,
                    k = 0L,
                    dtype = c("float32", "float64")) {
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
  local_float64_cpu(dtype)
  ptr <- cpp_mlx_tri(n, m_arg, k, dtype)
  new_mlx(ptr)
}

#' @rdname mlx_tri
#' @export
mlx_tril <- function(x, k = 0L) {
  x <- as_mlx(x)

  k <- as.integer(k)
  if (length(k) != 1L || is.na(k)) {
    stop("k must be a single integer.", call. = FALSE)
  }

  ptr <- cpp_mlx_tril(x$ptr, k)
  new_mlx(ptr)
}

#' @rdname mlx_tri
#' @export
mlx_triu <- function(x, k = 0L) {
  x <- as_mlx(x)

  k <- as.integer(k)
  if (length(k) != 1L || is.na(k)) {
    stop("k must be a single integer.", call. = FALSE)
  }

  ptr <- cpp_mlx_triu(x$ptr, k)
  new_mlx(ptr)
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

  ptr <- cpp_mlx_diag(x$ptr, k)
  new_mlx(ptr)
}

#' Numerical ranges on MLX devices
#'
#' `mlx_arange()` creates evenly spaced values starting at `start`, stepping by `step`,
#' up to and including `stop` (if exactly reachable). This matches R's [base::seq()] behavior.
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @param start Starting value.
#' @param stop Upper bound (included if exactly reachable by the step sequence).
#' @param step Step size (defaults to 1).
#' @inheritParams mlx_zeros
#' @return A 1D mlx array.
#' @seealso [mlx.core.arange](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.arange)
#'
#' @section Difference from Python/C++:
#' Unlike Python's `range()` and `numpy.arange()` which use an exclusive upper bound,
#' `mlx_arange()` matches R's [base::seq()] by including `stop` only if it's exactly
#' reachable by the step sequence. This is consistent with [mlx_linspace()] and
#' [mlx_slice_update()], which also follow R conventions.
#'
#' @export
#' @examples
#' mlx_arange(0, 4)        # 0, 1, 2, 3, 4
#' mlx_arange(1, 5)        # 1, 2, 3, 4, 5
#' mlx_arange(1, 9, 2)     # 1, 3, 5, 7, 9
#' mlx_arange(1, 6, 2)     # 1, 3, 5 (6 not reachable)
mlx_arange <- function(start,
                       stop,
                       step = 1,
                       dtype = c("float32", "float64", "int8", "int16", "int32", "int64",
                                "uint8", "uint16", "uint32", "uint64")) {
  if (length(start) != 1L) {
    stop("start must be a single numeric value.", call. = FALSE)
  }
  if (length(stop) != 1L) {
    stop("stop must be a single numeric value.", call. = FALSE)
  }
  if (length(step) != 1L) {
    stop("step must be a single numeric value.", call. = FALSE)
  }

  dtype <- match.arg(dtype)
  local_float64_cpu(dtype)

  # Convert to exclusive stop for underlying MLX function
  # Add a tiny epsilon to include stop if exactly reachable (like seq())
  # but not to include the next step beyond
  stop_exclusive <- if (step > 0) {
    stop + 1e-10
  } else {
    stop - 1e-10
  }

  ptr <- cpp_mlx_arange(as.numeric(start), as.numeric(stop_exclusive), as.numeric(step), dtype)
  new_mlx(ptr)
}

#' Evenly spaced ranges on MLX devices
#'
#' `mlx_linspace()` creates `num` evenly spaced values from `start` to `stop`, inclusive.
#' Unlike `mlx_arange()`, you specify how many samples you want rather than the step size.
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @param start Starting value.
#' @param stop Final value (inclusive).
#' @param num Number of samples to generate.
#' @inheritParams mlx_arange
#' @return A 1D mlx array.
#' @seealso [mlx.core.linspace](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.linspace)
#' @export
#' @examples
#' mlx_linspace(0, 1, num = 5)
mlx_linspace <- function(start,
                         stop,
                         num = 50L,
                         dtype = c("float32", "float64")) {
  if (length(num) != 1L || num <= 0) {
    stop("num must be a positive integer.", call. = FALSE)
  }

  dtype <- match.arg(dtype)
  local_float64_cpu(dtype)

  ptr <- cpp_mlx_linspace(
    as.numeric(start),
    as.numeric(stop),
    as.integer(num),
    dtype
  )
  new_mlx(ptr)
}

# Helper to validate shapes ----------------------------------------------------

#' Validate and coerce shape specification
#'
#' @param dim Integer or numeric vector of dimension sizes.
#' @return Integer vector of positive dimensions.
#' @noRd
validate_shape <- function(dim) {
  if (length(dim) == 0L) {
    return(integer(0))
  }
  dim <- as.integer(dim)
  if (any(is.na(dim) | dim <= 0)) {
    stop("dim must contain positive integers.", call. = FALSE)
  }
  dim
}
