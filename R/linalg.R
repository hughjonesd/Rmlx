#' Solve a system of linear equations
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @param a An mlx matrix of coefficients.
#' @param b An mlx vector or matrix (the right-hand side). If omitted,
#'   computes the matrix inverse.
#' @inheritParams ellipsis_base
#' @inheritParams common_params
#' @return An mlx object containing the solution.
#' @seealso [mlx.linalg.solve](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.solve)
#' @export
#' @examples
#' with_device("cpu", {
#'   a <- mlx_matrix(c(3, 1, 1, 2), 2, 2)
#'   b <- as_mlx(c(9, 8))
#'   solve(a, b)
#' })
solve.mlx <- function(a, b = NULL, ..., device = NULL) {
  with_optional_device(device, {
    a <- as_mlx(a)
    target_dtype <- mlx_dtype(a)
    if (!(target_dtype %in% c("float32", "float64", "complex64"))) {
      target_dtype <- "float32"
    }

    if (is.null(b)) {
      a <- mlx_cast(a, dtype = target_dtype)
      ptr <- cpp_mlx_solve(a$ptr, NULL, target_dtype)
    } else {
      if (!is_mlx(b)) {
        b <- as_mlx(b, dtype = target_dtype)
      } else {
        target_dtype <- resolve_common_dtype(list(target_dtype, mlx_dtype(b)))
        a <- mlx_cast(a, dtype = target_dtype)
        b <- mlx_cast(b, dtype = target_dtype)
      }

      ptr <- cpp_mlx_solve(a$ptr, b$ptr, target_dtype)
    }

    new_mlx(ptr)
  })
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
#' automatically cast to a common dtype before evaluation.
#'
#' @param a,b Objects coercible to `mlx`.
#' @return An `mlx` array representing the Kronecker product.
#' @seealso [mlx.core.kron](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.kron)
#' @export
#' @examples
#' A <- mlx_matrix(1:4, 2, 2)
#' B <- mlx_matrix(c(0, 5, 6, 7), 2, 2)
#' mlx_kron(A, B)
mlx_kron <- function(a, b) {
  operands <- coerce_binary_operands(a, b)
  a <- operands[[1L]]
  b <- operands[[2L]]

  target_dtype <- resolve_common_dtype(list(mlx_dtype(a), mlx_dtype(b)))

  a <- mlx_cast(a, dtype = target_dtype)
  b <- mlx_cast(b, dtype = target_dtype)

  ptr <- cpp_mlx_kron(a$ptr, b$ptr)
  new_mlx(ptr)
}

setOldClass("mlx")

#' @importFrom methods setMethod setOldClass
#' @importMethodsFrom methods kronecker
NULL

#' @export
#' @docType methods
#' @rdname kronecker
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

#' @export
#' @docType methods
#' @rdname kronecker
setMethod(
  "kronecker",
  signature(X = "mlx", Y = "ANY"),
  function(X, Y, ...) {
    mlx_kron(X, as_mlx(Y))
  }
)

#' @export
#' @docType methods
#' @rdname kronecker
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
#' @inheritParams ellipsis_base
#' @return An `mlx` array.
#' @rdname kronecker
#' @export
kronecker.mlx <- function(X, Y, FUN = "*", ..., make.dimnames = FALSE) {
  if (!identical(FUN, "*")) {
    stop("Only FUN='*' is supported for mlx kronecker.", call. = FALSE)
  }
  if (!identical(make.dimnames, FALSE)) {
    warning("make.dimnames is ignored for mlx results.", call. = FALSE)
  }
  if (!is_mlx(Y)) {
    Y <- as_mlx(Y)
  }
  mlx_kron(X, Y)
}
