#' Arithmetic and comparison operators for MLX arrays
#'
#' @param e1 First operand (mlx or numeric)
#' @param e2 Second operand (mlx or numeric)
#' @return An `mlx` object
#' @export
#' @method Ops mlx
Ops.mlx <- function(e1, e2 = NULL) {
  op <- .Generic

  # Unary operators
  if (nargs() == 1) {
    if (op == "+") return(e1)
    if (op == "-") return(.mlx_unary(e1, "neg"))
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

  stop(sprintf("Operator '%s' not supported for mlx", op))
}

#' Matrix multiplication for MLX arrays
#'
#' @inheritParams base::`%*%`
#' @return An `mlx` object
#' @export
#' @method %*% mlx
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

# Internal helper: unary operation
.mlx_unary <- function(x, op) {
  ptr <- cpp_mlx_unary(x$ptr, op)
  new_mlx(ptr, x$dim, x$dtype, x$device)
}

# Internal helper: binary operation
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

# Internal helper: broadcast dimensions
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

# Internal helper: promote dtype
.promote_dtype <- function(dtype1, dtype2) {
  if (dtype1 == dtype2) return(dtype1)

  dtypes <- c(dtype1, dtype2)

  if ("complex64" %in% dtypes) return("complex64")
  if ("float32" %in% dtypes) return("float32")
  if ("float64" %in% dtypes) return("float32")
  if ("bool" %in% dtypes) return("float32")

  stop("Unsupported dtype combination: ", dtype1, " and ", dtype2)
}

# Internal helper: common device
.common_device <- function(device1, device2) {
  if (device1 == device2) return(device1)
  # Prefer GPU if devices differ
  if (device1 == "gpu" || device2 == "gpu") return("gpu")
  return("cpu")
}
