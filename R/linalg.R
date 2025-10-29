#' Solve a system of linear equations
#'
#' @param a An mlx matrix (the coefficient matrix)
#' @param b An mlx vector or matrix (the right-hand side). If omitted,
#'   computes the matrix inverse.
#' @param ... Additional arguments (for compatibility with base::solve)
#' @return An mlx object containing the solution
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.solve}
#' @export
#' @method solve mlx
#' @examples
#' a <- as_mlx(matrix(c(3, 1, 1, 2), 2, 2))
#' b <- as_mlx(c(9, 8))
#' solve(a, b)
solve.mlx <- function(a, b = NULL, ...) {
  target_device <- a$device
  target_dtype <- "float32"

  if (is.null(b)) {
    # No b: compute matrix inverse
    ptr <- cpp_mlx_solve(a$ptr, NULL, target_dtype, target_device)
    new_mlx(ptr, a$dim, target_dtype, target_device)
  } else {
    # Convert b to mlx if needed
    if (!is.mlx(b)) {
      b <- as_mlx(b, dtype = target_dtype, device = target_device)
    }

    # Solve Ax = b
    ptr <- cpp_mlx_solve(a$ptr, b$ptr, target_dtype, target_device)

    # Result dimensions: if b is a vector, result is a vector
    # if b is a matrix with k columns, result has same dimensions as b
    new_mlx(ptr, b$dim, target_dtype, target_device)
  }
}

#' Kronecker product dispatcher
#'
#' Wrapper around [base::kronecker()] that enables S3 dispatch for `mlx` arrays
#' while delegating to base R for all other inputs.
#'
#' @inheritParams base::kronecker
#' @rdname kronecker
#' @export
kronecker <- function(X, Y, FUN = "*", make.dimnames = FALSE, ...) {
  UseMethod("kronecker")
}

#' @rdname kronecker
#' @export
kronecker.default <- function(X, Y, FUN = "*", make.dimnames = FALSE, ...) {
  base::kronecker(X, Y, FUN = FUN, make.dimnames = make.dimnames, ...)
}

#' Kronecker product for mlx arrays
#'
#' Computes the Kronecker (tensor) product between two mlx arrays. Inputs are
#' automatically cast to a common dtype and device before evaluation.
#'
#' @param a,b Objects coercible to `mlx`.
#' @return An `mlx` array representing the Kronecker product.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.kron}
#' @export
#' @examples
#' A <- as_mlx(matrix(1:4, 2, 2))
#' B <- as_mlx(matrix(c(0, 5, 6, 7), 2, 2))
#' mlx_kron(A, B)
mlx_kron <- function(a, b) {
  a <- as_mlx(a)
  b <- as_mlx(b)

  result_dtype <- .promote_dtype(a$dtype, b$dtype)
  result_device <- .common_device(a$device, b$device)

  a <- .mlx_cast(a, dtype = result_dtype, device = result_device)
  b <- .mlx_cast(b, dtype = result_dtype, device = result_device)

  ptr <- cpp_mlx_kron(a$ptr, b$ptr, result_device)
  out_dim <- cpp_mlx_shape(ptr)
  new_mlx(ptr, out_dim, result_dtype, result_device)
}

setOldClass("mlx")

setMethod(
  "kronecker",
  signature(X = "mlx", Y = "mlx"),
  function(X, Y, FUN = "*", make.dimnames = FALSE, ...) {
    if (!identical(FUN, "*")) {
      stop("Only FUN='*' is supported for mlx kronecker.", call. = FALSE)
    }
    if (!identical(make.dimnames, FALSE)) {
      warning("make.dimnames is ignored for mlx results.", call. = FALSE)
    }
    mlx_kron(X, Y)
  }
)

setMethod(
  "kronecker",
  signature(X = "mlx", Y = "ANY"),
  function(X, Y, ...) {
    mlx_kron(X, as_mlx(Y))
  }
)

setMethod(
  "kronecker",
  signature(X = "ANY", Y = "mlx"),
  function(X, Y, ...) {
    mlx_kron(as_mlx(X), Y)
  }
)

#' Kronecker method for mlx objects (S3 dispatch)
#'
#' Ensures the base `kronecker()` generic can dispatch on S3 `mlx` objects when
#' S4 dispatch is unavailable.
#'
#' @inheritParams mlx_kron
#' @param FUN Must be `'*'` (other functions are unsupported for MLX tensors).
#' @param ... Passed to maintain signature compatibility with base `kronecker()`.
#' @return An `mlx` array.
#' @rdname kronecker
#' @export
#' @method kronecker mlx
kronecker.mlx <- function(X, Y, FUN = "*", ..., make.dimnames = FALSE) {
  if (!identical(FUN, "*")) {
    stop("Only FUN='*' is supported for mlx kronecker.", call. = FALSE)
  }
  if (!identical(make.dimnames, FALSE)) {
    warning("make.dimnames is ignored for mlx results.", call. = FALSE)
  }
  if (!is.mlx(Y)) {
    Y <- as_mlx(Y)
  }
  mlx_kron(X, Y)
}
