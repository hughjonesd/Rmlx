#' Arithmetic and comparison operators for MLX arrays
#'
#' @param e1 First operand (mlx or numeric)
#' @param e2 Second operand (mlx or numeric)
#' @return An mlx object
#' @seealso [mlx.core.array](https://ml-explore.github.io/mlx/build/html/python/array.html)
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
      e1 <- as_mlx(e1)
      return(.mlx_logical_not(e1))
    }
    stop(sprintf("Unary operator '%s' not supported for mlx", op))
  }

  # Binary operators - coerce arguments to mlx
  e1 <- as_mlx(e1)
  e2 <- as_mlx(e2)

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
#' @return An mlx object
#' @seealso [mlx.core.matmul](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.matmul)
#' @export
#' @method %*% mlx
#' @examples
#' \dontrun{
#' x <- as_mlx(matrix(1:6, 2, 3))
#' y <- as_mlx(matrix(1:6, 3, 2))
#' x %*% y
#' }
`%*%.mlx` <- function(x, y) {
  x <- as_mlx(x)
  y <- as_mlx(y)

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

#' Fused matrix multiply and add for MLX arrays
#'
#' Computes `beta * input + alpha * (mat1 %*% mat2)` in a single MLX kernel.
#' All operands are promoted to a common dtype/device prior to evaluation.
#'
#' @param input Matrix-like object providing the additive term.
#' @param mat1 Left matrix operand.
#' @param mat2 Right matrix operand.
#' @param alpha,beta Numeric scalars controlling the fused linear combination.
#' @return An `mlx` matrix with the same shape as `input`.
#' @seealso [mlx.core.addmm](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.addmm)
#' @export
#' @examples
#' \dontrun{
#' input <- as_mlx(diag(3))
#' mat1 <- as_mlx(matrix(rnorm(9), 3, 3))
#' mat2 <- as_mlx(matrix(rnorm(9), 3, 3))
#' mlx_addmm(input, mat1, mat2, alpha = 0.5, beta = 2)
#' }
mlx_addmm <- function(input, mat1, mat2, alpha = 1, beta = 1) {
  input <- as_mlx(input)
  mat1 <- as_mlx(mat1)
  mat2 <- as_mlx(mat2)

  dims_input <- input$dim
  dims1 <- mat1$dim
  dims2 <- mat2$dim

  if (length(dims1) != 2L || length(dims2) != 2L) {
    stop("mlx_addmm requires mat1 and mat2 to be 2D matrices.", call. = FALSE)
  }
  if (length(dims_input) != 2L) {
    stop("mlx_addmm requires input to be a 2D matrix.", call. = FALSE)
  }
  if (dims1[2] != dims2[1]) {
    stop(
      sprintf(
        "Non-conformable operands: mat1 is %d x %d but mat2 is %d x %d.",
        dims1[1], dims1[2], dims2[1], dims2[2]
      ),
      call. = FALSE
    )
  }
  result_dim <- c(dims1[1], dims2[2])
  if (!identical(dims_input, result_dim)) {
    stop(
      sprintf(
        "Input shape (%d x %d) must match mat1 %%*%% mat2 result (%d x %d).",
        dims_input[1], dims_input[2], result_dim[1], result_dim[2]
      ),
      call. = FALSE
    )
  }

  alpha <- as.numeric(alpha)
  beta <- as.numeric(beta)
  if (length(alpha) != 1L || !is.finite(alpha)) {
    stop("alpha must be a single finite numeric value.", call. = FALSE)
  }
  if (length(beta) != 1L || !is.finite(beta)) {
    stop("beta must be a single finite numeric value.", call. = FALSE)
  }

  result_dtype <- Reduce(.promote_dtype, list(input$dtype, mat1$dtype, mat2$dtype))
  result_device <- Reduce(.common_device, list(input$device, mat1$device, mat2$device))

  input <- .mlx_cast(input, dtype = result_dtype, device = result_device)
  mat1 <- .mlx_cast(mat1, dtype = result_dtype, device = result_device)
  mat2 <- .mlx_cast(mat2, dtype = result_dtype, device = result_device)

  ptr <- cpp_mlx_addmm(input$ptr, mat1$ptr, mat2$ptr, alpha, beta, result_dtype, result_device)
  new_mlx(ptr, result_dim, result_dtype, result_device)
}

#' Apply unary MLX operation
#'
#' @inheritParams mlx_array_required
#' @param op Character string naming the operation.
#' @return mlx array with same shape.
#' @noRd
.mlx_unary <- function(x, op) {
  ptr <- cpp_mlx_unary(x$ptr, op)
  new_mlx(ptr, x$dim, x$dtype, x$device)
}

#' Apply binary MLX operation with type promotion
#'
#' @param x,y mlx arrays.
#' @param op Character string naming the operation.
#' @return mlx array with broadcasted dimensions.
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

#' Apply logical MLX operation
#'
#' @param x,y mlx arrays.
#' @param op Character string naming the logical operation.
#' @return mlx array with dtype "bool".
#' @noRd
.mlx_logical <- function(x, y, op) {
  result_dim <- .broadcast_dim(x$dim, y$dim)
  result_device <- .common_device(x$device, y$device)

  ptr <- cpp_mlx_logical(x$ptr, y$ptr, op, result_device)
  new_mlx(ptr, result_dim, "bool", result_device)
}

#' Apply logical NOT to mlx array
#'
#' @inheritParams mlx_array_required
#' @return mlx array with dtype "bool".
#' @noRd
.mlx_logical_not <- function(x) {
  ptr <- cpp_mlx_logical_not(x$ptr)
  new_mlx(ptr, x$dim, "bool", x$device)
}

#' Integer division for mlx arrays
#'
#' @param x,y mlx arrays.
#' @return mlx array with floor-divided values.
#' @noRd
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

#' Remainder operation for mlx arrays
#'
#' @param x,y mlx arrays.
#' @return mlx array with remainder values.
#' @noRd
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

#' Elementwise minimum of two mlx arrays
#'
#' @param x,y mlx arrays or objects coercible with [as_mlx()].
#' @return An mlx array containing the elementwise minimum.
#' @seealso [mlx.core.minimum](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.minimum)
#' @export
#' @examples
#' \dontrun{
#' a <- as_mlx(matrix(1:4, 2, 2))
#' b <- as_mlx(matrix(c(4, 3, 2, 1), 2, 2))
#' mlx_minimum(a, b)
#' }
mlx_minimum <- function(x, y) {
  .mlx_binary_result(x, y, cpp_mlx_minimum)
}

#' Elementwise maximum of two mlx arrays
#'
#' @inheritParams mlx_minimum
#' @return An mlx array containing the elementwise maximum.
#' @seealso [mlx.core.maximum](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.maximum)
#' @export
#' @examples
#' \dontrun{
#' mlx_maximum(1:3, c(3, 2, 1))
#' }
mlx_maximum <- function(x, y) {
  .mlx_binary_result(x, y, cpp_mlx_maximum)
}

#' Internal wrapper for binary operations
#'
#' @param x,y mlx arrays or coercible to mlx
#' @param cpp_fn C++ function to call (takes x$ptr, y$ptr, device)
#' @return mlx array
#' @noRd
.mlx_binary_result <- function(x, y, cpp_fn) {
  x <- as_mlx(x)
  y <- as_mlx(y)

  result_dim <- .broadcast_dim(x$dim, y$dim)
  result_device <- .common_device(x$device, y$device)
  result_dtype <- .promote_dtype(x$dtype, y$dtype)

  if (identical(result_dtype, "bool")) {
    result_dtype <- "float32"
  }

  ptr <- cpp_fn(x$ptr, y$ptr, result_device)
  new_mlx(ptr, result_dim, result_dtype, result_device)
}

#' Clip mlx array values into a range
#'
#' @inheritParams common_params
#' @param min,max Scalar bounds. Use `NULL` to leave a bound open.
#' @return An mlx array with values clipped to `[min, max]`.
#' @seealso [mlx.core.clip](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.clip)
#' @export
#' @examples
#' \dontrun{
#' x <- as_mlx(rnorm(4))
#' mlx_clip(x, min = -1, max = 1)
#' }
mlx_clip <- function(x, min = NULL, max = NULL) {
  x <- as_mlx(x)
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

#' Compute broadcast dimensions
#'
#' @param dim1,dim2 Integer vectors of dimensions.
#' @return Integer vector of broadcasted dimensions.
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

#' Promote dtypes for mixed operations
#'
#' @param dtype1,dtype2 Character strings naming dtypes.
#' @return Character string of promoted dtype.
#' @noRd
.promote_dtype <- function(dtype1, dtype2) {
  if (dtype1 == dtype2) return(dtype1)

  dtypes <- c(dtype1, dtype2)

  # Complex beats everything
  if ("complex64" %in% dtypes) return("complex64")

  # Float beats integer and bool
  if ("float64" %in% dtypes || "float32" %in% dtypes) return("float32")

  # Bool promotes to int32
  if ("bool" %in% dtypes) {
    other <- setdiff(dtypes, "bool")[1]
    if (other %in% c("int8", "int16", "int32", "int64",
                     "uint8", "uint16", "uint32", "uint64")) {
      return(other)
    }
    return("int32")
  }

  # Integer type promotion - promote to larger/wider type
  int_types <- c("int8", "int16", "int32", "int64",
                 "uint8", "uint16", "uint32", "uint64")

  if (all(dtypes %in% int_types)) {
    # Type hierarchy (smaller to larger)
    # For simplicity, promote to int64 for mixed signed/unsigned of same size
    type_rank <- c(int8 = 1, uint8 = 1, int16 = 2, uint16 = 2,
                   int32 = 3, uint32 = 3, int64 = 4, uint64 = 4)

    rank1 <- type_rank[dtype1]
    rank2 <- type_rank[dtype2]

    # Promote to the larger rank
    if (rank1 > rank2) return(dtype1)
    if (rank2 > rank1) return(dtype2)

    # Same rank but different signedness -> promote to signed of next size
    # e.g., int32 + uint32 -> int64
    if (rank1 == rank2 && dtype1 != dtype2) {
      next_signed <- c("1" = "int16", "2" = "int32", "3" = "int64", "4" = "int64")
      return(next_signed[as.character(rank1)])
    }
  }

  stop("Unsupported dtype combination: ", dtype1, " and ", dtype2)
}

#' Select common device from two arrays
#'
#' @param device1,device2 Character strings ("gpu" or "cpu").
#' @return Character string ("gpu" or "cpu").
#' @noRd
.common_device <- function(device1, device2) {
  if (device1 == device2) return(device1)
  # Prefer GPU if devices differ
  if (device1 == "gpu" || device2 == "gpu") return("gpu")
  return("cpu")
}
