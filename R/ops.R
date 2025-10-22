#' Arithmetic and comparison operators for MLX arrays
#'
#' @param e1 First operand (mlx or numeric)
#' @param e2 Second operand (mlx or numeric)
#' @return An `mlx` object
#' @export
#' @method Ops mlx
#' @examples
#' \dontrun{
#' x <- as_mlx(matrix(1:4, 2, 2))
#' y <- as_mlx(matrix(5:8, 2, 2))
#' x + y
#' x < y
#' }
Ops.mlx <- function(e1, e2 = NULL) {
  op <- .Generic

  # Unary operators
  if (nargs() == 1) {
    if (op == "+") return(e1)
    if (op == "-") return(.mlx_unary(e1, "neg"))
    if (op == "!") {
      if (!is.mlx(e1)) e1 <- as_mlx(e1)
      return(.mlx_logical_not(e1))
    }
    stop(sprintf("Unary operator '%s' not supported for mlx", op))
  }

  # Binary operators - coerce arguments to mlx
  if (!is.mlx(e1)) e1 <- as_mlx(e1)
  if (!is.mlx(e2)) e2 <- as_mlx(e2)

  # Arithmetic operators
  if (op %in% c("+", "-", "*", "/", "^")) {
    return(.mlx_binary(e1, e2, op))
  }

  # Comparison operators
  if (op %in% c("==", "!=", "<", "<=", ">", ">=")) {
    return(.mlx_binary(e1, e2, op))
  }

  # Modulo / floor division
  if (op == "%/%") {
    return(.mlx_floor_divide(e1, e2))
  }
  if (op == "%%") {
    return(.mlx_remainder(e1, e2))
  }

  # Logical operators
  if (op %in% c("&", "&&", "|", "||")) {
    return(.mlx_logical(e1, e2, op))
  }

  stop(sprintf("Operator '%s' not supported for mlx", op))
}

#' Matrix multiplication for MLX arrays
#'
#' @inheritParams base::`%*%`
#' @return An `mlx` object
#' @export
#' @method %*% mlx
#' @examples
#' \dontrun{
#' x <- as_mlx(matrix(1:6, 2, 3))
#' y <- as_mlx(matrix(1:6, 3, 2))
#' x %*% y
#' }
`%*%.mlx` <- function(x, y) {
  if (!is.mlx(x)) x <- as_mlx(x)
  if (!is.mlx(y)) y <- as_mlx(y)

  # Validate dimensions
  if (length(x$dim) != 2 || length(y$dim) != 2) {
    stop("Matrix multiplication requires 2D arrays")
  }

  if (x$dim[2] != y$dim[1]) {
    stop(sprintf(
      "Non-conformable arrays: %d x %d and %d x %d",
      x$dim[1], x$dim[2], y$dim[1], y$dim[2]
    ))
  }

  result_dim <- c(x$dim[1], y$dim[2])
  result_dtype <- .promote_dtype(x$dtype, y$dtype)
  result_device <- .common_device(x$device, y$device)

  ptr <- cpp_mlx_matmul(x$ptr, y$ptr, result_dtype, result_device)
  new_mlx(ptr, result_dim, result_dtype, result_device)
}

#' Apply a unary MLX kernel
#'
#' @param x An `mlx` tensor.
#' @param op Operation name forwarded to C++.
#' @return Updated `mlx` tensor with the same shape.
#' @noRd
.mlx_unary <- function(x, op) {
  ptr <- cpp_mlx_unary(x$ptr, op)
  new_mlx(ptr, x$dim, x$dtype, x$device)
}

#' Apply a binary MLX kernel with dtype/device alignment
#'
#' @param x,y `mlx` tensors to combine.
#' @param op Operation identifier.
#' @return Resulting `mlx` tensor.
#' @noRd
.mlx_binary <- function(x, y, op) {
  result_dim <- .broadcast_dim(x$dim, y$dim)
  input_dtype <- .promote_dtype(x$dtype, y$dtype)
  result_device <- .common_device(x$device, y$device)

  is_comparison <- op %in% c("==", "!=", "<", "<=", ">", ">=")

  if (!is_comparison && identical(input_dtype, "bool")) {
    input_dtype <- "float32"
  }

  result_dtype <- if (is_comparison) "bool" else input_dtype

  ptr <- cpp_mlx_binary(x$ptr, y$ptr, op, input_dtype, result_device)
  new_mlx(ptr, result_dim, result_dtype, result_device)
}

.mlx_logical <- function(x, y, op) {
  result_dim <- .broadcast_dim(x$dim, y$dim)
  result_device <- .common_device(x$device, y$device)

  ptr <- cpp_mlx_logical(x$ptr, y$ptr, op, result_device)
  new_mlx(ptr, result_dim, "bool", result_device)
}

.mlx_logical_not <- function(x) {
  ptr <- cpp_mlx_logical_not(x$ptr)
  new_mlx(ptr, x$dim, "bool", x$device)
}

.mlx_floor_divide <- function(x, y) {
  result_dim <- .broadcast_dim(x$dim, y$dim)
  result_device <- .common_device(x$device, y$device)
  result_dtype <- .promote_dtype(x$dtype, y$dtype)

  if (identical(result_dtype, "bool")) {
    result_dtype <- "float32"
  }

  ptr <- cpp_mlx_floor_divide(x$ptr, y$ptr, result_device)
  new_mlx(ptr, result_dim, result_dtype, result_device)
}

.mlx_remainder <- function(x, y) {
  result_dim <- .broadcast_dim(x$dim, y$dim)
  result_device <- .common_device(x$device, y$device)
  result_dtype <- .promote_dtype(x$dtype, y$dtype)

  if (identical(result_dtype, "bool")) {
    result_dtype <- "float32"
  }

  ptr <- cpp_mlx_remainder(x$ptr, y$ptr, result_device)
  new_mlx(ptr, result_dim, result_dtype, result_device)
}

#' Elementwise minimum of two MLX tensors
#'
#' @param x,y `mlx` tensors or objects coercible with [as_mlx()].
#' @return An `mlx` tensor containing the elementwise minimum.
#' @export
#' @examples
#' \dontrun{
#' a <- as_mlx(matrix(1:4, 2, 2))
#' b <- as_mlx(matrix(c(4, 3, 2, 1), 2, 2))
#' mlx_minimum(a, b)
#' }
mlx_minimum <- function(x, y) {
  if (!is.mlx(x)) x <- as_mlx(x)
  if (!is.mlx(y)) y <- as_mlx(y)

  result_dim <- .broadcast_dim(x$dim, y$dim)
  result_device <- .common_device(x$device, y$device)
  result_dtype <- .promote_dtype(x$dtype, y$dtype)

  if (identical(result_dtype, "bool")) {
    result_dtype <- "float32"
  }

  ptr <- cpp_mlx_minimum(x$ptr, y$ptr, result_device)
  new_mlx(ptr, result_dim, result_dtype, result_device)
}

#' Elementwise maximum of two MLX tensors
#'
#' @inheritParams mlx_minimum
#' @return An `mlx` tensor containing the elementwise maximum.
#' @export
#' @examples
#' \dontrun{
#' mlx_maximum(1:3, c(3, 2, 1))
#' }
mlx_maximum <- function(x, y) {
  if (!is.mlx(x)) x <- as_mlx(x)
  if (!is.mlx(y)) y <- as_mlx(y)

  result_dim <- .broadcast_dim(x$dim, y$dim)
  result_device <- .common_device(x$device, y$device)
  result_dtype <- .promote_dtype(x$dtype, y$dtype)

  if (identical(result_dtype, "bool")) {
    result_dtype <- "float32"
  }

  ptr <- cpp_mlx_maximum(x$ptr, y$ptr, result_device)
  new_mlx(ptr, result_dim, result_dtype, result_device)
}

#' Clip MLX tensor values into a range
#'
#' @param x An `mlx` tensor or coercible object.
#' @param min,max Scalar bounds. Use `NULL` to leave a bound open.
#' @return An `mlx` tensor with values clipped to `[min, max]`.
#' @export
#' @examples
#' \dontrun{
#' x <- as_mlx(rnorm(4))
#' mlx_clip(x, min = -1, max = 1)
#' }
mlx_clip <- function(x, min = NULL, max = NULL) {
  if (!is.mlx(x)) x <- as_mlx(x)
  if (is.null(min) && is.null(max)) {
    stop("At least one of 'min' or 'max' must be supplied.", call. = FALSE)
  }
  if (!is.null(min) && length(min) != 1L) {
    stop("'min' must be NULL or a scalar.", call. = FALSE)
  }
  if (!is.null(max) && length(max) != 1L) {
    stop("'max' must be NULL or a scalar.", call. = FALSE)
  }

  ptr <- cpp_mlx_clip(x$ptr, min, max, x$device)
  new_mlx(ptr, x$dim, if (x$dtype %in% c("float32", "float64")) x$dtype else "float32", x$device)
}

#' Broadcast two dimension vectors
#'
#' @param dim1,dim2 Dimension vectors.
#' @return Broadcasted dimensions.
#' @noRd
.broadcast_dim <- function(dim1, dim2) {
  # Simplified broadcasting rules
  # In reality, this should follow NumPy-style broadcasting
  if (identical(dim1, dim2)) {
    return(dim1)
  }

  # If one is a scalar (length 1), use the other's dimensions
  if (prod(dim1) == 1) return(dim2)
  if (prod(dim2) == 1) return(dim1)

  # For now, assume dimensions match or one broadcasts
  # A more complete implementation would handle general broadcasting
  if (length(dim1) >= length(dim2)) {
    return(dim1)
  } else {
    return(dim2)
  }
}

#' Promote two MLX dtypes to a computation dtype
#'
#' @param dtype1,dtype2 Character dtype names.
#' @return Promoted dtype.
#' @noRd
.promote_dtype <- function(dtype1, dtype2) {
  if (dtype1 == dtype2) return(dtype1)

  dtypes <- c(dtype1, dtype2)

  if ("complex64" %in% dtypes) return("complex64")
  if ("float32" %in% dtypes) return("float32")
  if ("float64" %in% dtypes) return("float32")
  if ("bool" %in% dtypes) return("float32")

  stop("Unsupported dtype combination: ", dtype1, " and ", dtype2)
}

#' Choose a common device for two tensors
#'
#' @param device1,device2 Device strings.
#' @return Device string.
#' @noRd
.common_device <- function(device1, device2) {
  if (device1 == device2) return(device1)
  # Prefer GPU if devices differ
  if (device1 == "gpu" || device2 == "gpu") return("gpu")
  return("cpu")
}
