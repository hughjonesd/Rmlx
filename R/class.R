#' Coerce R payload into the storage format expected by MLX
#'
#' @param x Input object (vector/array).
#' @param dtype Target MLX dtype.
#' @return Vector of numeric or complex values (or the original numeric input when
#'   it is already a double vector/matrix headed for float32/float64).
#' @noRd
.mlx_coerce_payload <- function(x, dtype) {
  if (dtype %in% c("float32", "float64") &&
      is.double(x) &&
      is.atomic(x) &&
      !is.object(x)) {
    return(x)
  }

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
#' @inheritParams common_params
#' @param x Numeric, logical, or complex vector, matrix, or array to convert
#' @param dtype Data type for the MLX array. One of:
#'   - Floating point: `"float32"`, `"float64"`
#'   - Integer signed: `"int8"`, `"int16"`, `"int32"`, `"int64"`
#'   - Integer unsigned: `"uint8"`, `"uint16"`, `"uint32"`, `"uint64"`
#'   - Other: `"bool"`, `"complex64"`
#'
#'   If not specified, defaults to `"float32"` for numeric, `"bool"` for logical,
#'   and `"complex64"` for complex inputs.
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
#' @seealso [mlx.core.array](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.array)
#' @seealso [mlx-methods]
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
  handle <- .mlx_resolve_device(device, mlx_default_device())
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

  dim_vec <- {
    dims <- dim(x)
    if (!is.null(dims)) {
      as.integer(dims)
    } else if (length(x) == 1L) {
      integer(0)
    } else {
      as.integer(length(x))
    }
  }
  x_payload <- .mlx_coerce_payload(x, dtype_val)

  # Create MLX array via C++
  ptr <- .mlx_eval_with_stream(handle, function(dev) {
    cpp_mlx_from_r(x_payload, as.integer(dim_vec), dtype_val, dev)
  })

  # Create S3 object (dim is always read from MLX via dim.mlx())
  new_mlx(ptr, handle$device)
}

#' Force evaluation of lazy MLX operations
#'
#' @inheritParams mlx_array_required
#' @return The input object (invisibly)
#' @seealso [mlx.core.eval](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.eval)
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
  if (length(dim(x)) == 0) {
    return(as.vector(out))
  }
  dim(out) <- dim(x)
  attrs <- attributes(x)
  attrs$names <- NULL
  attrs$class <- NULL
  if (length(attrs)) {
    for (nm in names(attrs)) {
      attr(out, nm) <- attrs[[nm]]
    }
  }
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
#' Converts an MLX array to an R vector. For multi-dimensional arrays (2+ dimensions),
#' the array is flattened in column-major order (R's default).
#'
#' @inheritParams mlx_array_required
#' @param mode Character string specifying the type of vector to return (passed to [base::as.vector()])
#' @return A vector of the specified mode
#' @export
#' @examples
#' x <- as_mlx(1:5)
#' as.vector(x)
#'
#' # Multi-dimensional arrays are flattened
#' m <- as_mlx(matrix(1:6, 2, 3))
#' v <- as.vector(m)  # Flattened in column-major order
as.vector.mlx <- function(x, mode = "any") {
  as.vector(as.matrix(x), mode = mode)
}

#' Convert MLX array to logical vector
#'
#' @inheritParams mlx_array_required
#' @param ... Additional arguments passed to [base::as.vector()].
#' @return A logical vector.
#' @export
#' @examples
#' x <- as_mlx(c(1, 0, 2))
#' as.logical(x)
as.logical.mlx <- function(x, ...) {
  as.logical(as.vector(x))
}

#' Convert MLX array to numeric vector
#'
#' Converts an MLX array to a numeric (double) vector, dropping dimensions
#' and coercing types as needed. Integers and booleans are converted to doubles.
#'
#' @inheritParams mlx_array_required
#' @param ... Additional arguments (currently unused).
#' @return A numeric vector.
#' @export
#' @examples
#' x_int <- as_mlx(c(1L, 2L, 3L), dtype = "int32")
#' as.numeric(x_int)  # Returns c(1.0, 2.0, 3.0)
#'
#' x_bool <- as_mlx(c(TRUE, FALSE, TRUE))
#' as.numeric(x_bool)  # Returns c(1.0, 0.0, 1.0)
as.double.mlx <- function(x, ...) {
  as.double(as.matrix(x))
}

#' @rdname as.double.mlx
#' @export
as.numeric.mlx <- as.double.mlx

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
#' @param device Device
#' @keywords internal
#' @noRd
new_mlx <- function(ptr, device) {
  structure(
    list(
      ptr = ptr,
      device = device
    ),
    class = "mlx"
  )
}
