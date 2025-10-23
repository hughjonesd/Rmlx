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
#' @param dtype Ignored. Present for backward compatibility. Numeric arrays are
#'   stored as `float32`; logical arrays use MLX `bool`; complex arrays use MLX
#'   `complex64`.
#' @param device Device: "gpu" (default) or "cpu"
#' @return An object of class `mlx`
#' @details Apple MLX executes in single precision. Numeric inputs are stored in
#'   `float32` regardless of the requested dtype. Logical inputs are mapped to
#'   MLX boolean tensors. Complex inputs are stored as `complex64` (float32 real
#'   and imaginary parts). Asking for `dtype = "float64"` emits a warning and the
#'   input is downcast to `float32`. If you require double precision arithmetic,
#'   use base R arrays instead of `mlx` objects.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.array}
#' @export
#' @examples
#' x <- as_mlx(matrix(1:12, 3, 4))
as_mlx <- function(x, dtype = c("float32", "float64", "bool", "complex64"), device = mlx_default_device()) {
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
#' @param x An `mlx` object
#' @return The input object (invisibly)
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
#' @param x An `mlx` object
#' @param ... Additional arguments (ignored)
#' @return A matrix or array (numeric or logical depending on dtype)
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
#' @param x An `mlx` object
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
#' @param x An `mlx` object
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
