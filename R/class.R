#' Infer dimensions for MLX conversion
#'
#' @param x Input object destined for MLX conversion.
#' @return Integer vector of dimensions (possibly length zero).
#' @noRd
.mlx_infer_dim <- function(x) {
  dims <- dim(x)
  if (!is.null(dims)) {
    return(as.integer(dims))
  }
  if (length(x) == 1L && is.null(dims)) {
    return(integer(0))
  }
  as.integer(length(x))
}

#' Coerce R payload into the storage format expected by MLX
#'
#' @param x Input object (vector/array).
#' @param dtype Target MLX dtype.
#' @return Vector of numeric or complex values.
#' @noRd
.mlx_coerce_payload <- function(x, dtype) {
  switch(
    dtype,
    "bool" = {
      x_logical <- as.logical(x)
      if (any(is.na(x_logical))) {
        stop("Logical NA values are not supported for MLX boolean arrays.", call. = FALSE)
      }
      as.numeric(x_logical)
    },
    "complex64" = as.complex(x),
    as.numeric(x)
  )
}

#' Create MLX array from R object
#'
#' @param x Numeric, logical, or complex vector, matrix, or array to convert
#' @param dtype Data type for the MLX array. One of:
#'   - Floating point: `"float32"`, `"float64"`
#'   - Integer signed: `"int8"`, `"int16"`, `"int32"`, `"int64"`
#'   - Integer unsigned: `"uint8"`, `"uint16"`, `"uint32"`, `"uint64"`
#'   - Other: `"bool"`, `"complex64"`
#'
#'   If not specified, defaults to `"float32"` for numeric, `"bool"` for logical,
#'   and `"complex64"` for complex inputs.
#' @param device Device: "gpu" (default) or "cpu"
#' @return An object of class `mlx`
#' @details
#' ## Default type behavior
#'
#' When `dtype` is not specified:
#' - Numeric vectors/arrays (including R integers from `1:10`) → `float32`
#' - Logical vectors/arrays → `bool`
#' - Complex vectors/arrays → `complex64`
#'
#' ## Integer types require explicit dtype
#'
#' **Important**: R integer vectors (like `1:10`) convert to `float32` by default.
#' To create integer MLX arrays, you must explicitly specify `dtype`:
#'
#' ```r
#' x <- as_mlx(1:10, dtype = "int32")  # Creates int32 array
#' x <- as_mlx(1:10)                    # Creates float32 array
#' ```
#'
#' This design avoids unintentional integer promotion, since R creates integers
#' in many contexts where floating-point is intended.
#'
#' ## Supported integer types
#'
#' - **Signed**: `int8` (-128 to 127), `int16`, `int32`, `int64`
#' - **Unsigned**: `uint8` (0 to 255), `uint16`, `uint32`, `uint64`
#'
#' ## Type precision notes
#'
#' - `float64` is supported but emits a warning and downcasts to `float32`
#' - Integer arithmetic may promote types (e.g., int32 + int32 might → int64)
#' - Mixed integer/float operations promote to float
#'
#' ## Missing values
#'
#' MLX does not have an `NA` sentinel. When you pass numeric `NA` values from R,
#' they are stored as `NaN` inside MLX and returned to R as `NaN`.
#' Use [is.nan()] on MLX arrays (method provided) if you need to detect them.
#'
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.array}
#' @export
#' @examples
#' # Default float32 for numeric
#' x <- as_mlx(c(1.5, 2.5, 3.5))
#' mlx_dtype(x)  # "float32"
#'
#' # R integers also default to float32
#' x <- as_mlx(1:10)
#' mlx_dtype(x)  # "float32"
#'
#' # Explicit integer types
#' x_int <- as_mlx(1:10, dtype = "int32")
#' mlx_dtype(x_int)  # "int32"
#'
#' # Unsigned integers
#' x_uint <- as_mlx(c(0, 128, 255), dtype = "uint8")
#'
#' # Logical → bool
#' mask <- as_mlx(c(TRUE, FALSE, TRUE))
#' mlx_dtype(mask)  # "bool"
as_mlx <- function(x, dtype = c("float32", "float64", "bool", "complex64",
                                 "int8", "int16", "int32", "int64",
                                 "uint8", "uint16", "uint32", "uint64"), device = mlx_default_device()) {
  device <- match.arg(device, c("gpu", "cpu"))
  dtype_val <- if (missing(dtype)) {
    if (is.logical(x)) {
      "bool"
    } else if (is.complex(x)) {
      "complex64"
    } else {
      "float32"
    }
  } else {
    match.arg(dtype)
  }

  if (dtype_val == "float64") {
    warning("MLX arrays are stored in float32; downcasting input.", call. = FALSE)
    dtype_val <- "float32"
  }

  if (is.mlx(x)) return(x)

  is_supported <- (is.vector(x) && !is.list(x)) || is.matrix(x) || is.array(x)
  if (!is_supported) {
    stop("Cannot convert object of class ", class(x)[1], " to mlx")
  }

  dim_vec <- .mlx_infer_dim(x)
  x_payload <- .mlx_coerce_payload(x, dtype_val)

  # Create MLX array via C++
  ptr <- cpp_mlx_from_r(x_payload, as.integer(dim_vec), dtype_val, device)

  # Create S3 object
  structure(
    list(
      ptr = ptr,
      dim = as.integer(dim_vec),
      dtype = dtype_val,
      device = device
    ),
    class = "mlx"
  )
}

#' Force evaluation of lazy MLX operations
#'
#' @inheritParams mlx_array_required
#' @return The input object (invisibly)
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.eval}
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_eval(x)
mlx_eval <- function(x) {
  stopifnot(is.mlx(x))
  cpp_mlx_eval(x$ptr)
  invisible(x)
}

#' Convert MLX array to R matrix/array
#'
#' MLX arrays without dimension are returned as R vectors.
#'
#' @inheritParams mlx_array_required
#' @param ... Additional arguments (ignored)
#' @return A vector, matrix or array (numeric or logical depending on dtype)
#' @export
#' @method as.matrix mlx
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' as.matrix(x)
as.matrix.mlx <- function(x, ...) {
  mlx_eval(x)
  out <- cpp_mlx_to_r(x$ptr)
  if (length(x$dim) == 0) {
    return(as.vector(out))
  }
  dim(out) <- x$dim
  out
}

#' Convert MLX array to R array
#'
#' @inheritParams mlx_array_required
#' @param ... Additional arguments (ignored)
#' @return A numeric array
#' @export
#' @examples
#' x <- as_mlx(matrix(1:8, 2, 4))
#' as.array(x)
as.array.mlx <- function(x, ...) {
  as.matrix.mlx(x, ...)
}

#' Convert MLX array to R vector
#'
#' @inheritParams mlx_array_required
#' @param mode Character string specifying the mode (ignored)
#' @return A numeric vector
#' @export
#' @examples
#' x <- as_mlx(1:5)
#' as.vector(x)
as.vector.mlx <- function(x, mode = "any") {
  # Only works for 1D arrays
  if (length(x$dim) == 0) {
    return(as.vector(as.matrix(x)))
  }

  if (length(x$dim) != 1) {
    stop("Cannot convert multi-dimensional mlx array to vector. Use as.vector(as.matrix(x)) to flatten.")
  }

  as.vector(as.matrix(x))
}

#' Test if object is an MLX array
#'
#' @param x Object to test
#' @return Logical
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' is.mlx(x)
is.mlx <- function(x) {
  inherits(x, "mlx")
}

#' Internal constructor for mlx objects
#'
#' @param ptr External pointer to MLX array
#' @param dim Dimensions
#' @param dtype Data type
#' @param device Device
#' @keywords internal
#' @noRd
new_mlx <- function(ptr, dim, dtype, device) {
  structure(
    list(
      ptr = ptr,
      dim = as.integer(dim),
      dtype = dtype,
      device = device
    ),
    class = "mlx"
  )
}
