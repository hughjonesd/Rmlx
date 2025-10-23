#' Wrap a raw MLX pointer into an `mlx` object
#'
#' @param ptr External pointer returned by C++ bindings.
#' @param device Device string associated with the tensor.
#' @return An `mlx` tensor.
#' @noRd
.mlx_wrap_result <- function(ptr, device) {
  dim <- cpp_mlx_shape(ptr)
  dtype <- cpp_mlx_dtype(ptr)
  new_mlx(ptr, dim, dtype, device)
}

#' Cholesky decomposition for MLX tensors
#'
#'
#' @param x An `mlx` matrix. Note: if `x` is not symmetric
#'   positive semi-definite, "behaviour is undefined" according to the MLX
#'   documentation.
#' @param pivot Ignored; pivoted decomposition is not supported.
#' @param ... Additional arguments (unused).
#' @return Upper-triangular Cholesky factor as an `mlx` matrix.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.cholesky}
#' @export
#' @method chol mlx
#' @examples
#' x <- as_mlx(matrix(c(4, 1, 1, 3), 2, 2))
#' chol(x)
chol.mlx <- function(x, pivot = FALSE, ...) {
  if (!is.mlx(x)) x <- as_mlx(x)
  if (pivot) stop("pivoted Cholesky is not supported for mlx objects.", call. = FALSE)

  ptr <- cpp_mlx_cholesky(x$ptr, TRUE, x$dtype, x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' QR decomposition for MLX tensors
#'
#' @param x An `mlx` matrix.
#' @param tol Ignored; custom tolerances are not supported.
#' @param LAPACK Ignored; set to `FALSE`.
#' @param ... Additional arguments (unused).
#' @return A list with components `Q` and `R`, each an `mlx` matrix.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.qr}
#' @export
#' @method qr mlx
#' @examples
#' x <- as_mlx(matrix(c(1, 2, 3, 4, 5, 6), 3, 2))
#' qr(x)
qr.mlx <- function(x, tol = 1e-7, LAPACK = FALSE, ...) {
  if (!is.mlx(x)) x <- as_mlx(x)
  stopifnot(length(x$dim) == 2L)

  if (!missing(tol) && !isTRUE(all.equal(tol, 1e-7))) {
    stop("Custom tolerance is not supported for mlx QR decomposition.", call. = FALSE)
  }
  if (LAPACK) stop("LAPACK = TRUE is not supported for mlx objects.", call. = FALSE)

  res <- cpp_mlx_qr(x$ptr, x$dtype, x$device)
  device <- x$device
  structure(
    list(
      Q = .mlx_wrap_result(res$Q, device),
      R = .mlx_wrap_result(res$R, device)
    ),
    class = c("mlx_qr", "list")
  )
}

#' @export
svd <- function(x, ...) {
  UseMethod("svd")
}

#' @export
svd.default <- base::svd

#' Singular value decomposition for MLX tensors
#'
#' Note that mlx's svd returns "full" SVD, with U and V' both square matrices.
#' This is different from R's implementation.
#'
#' @param x An `mlx` matrix.
#' @param nu Number of left singular vectors to return (0 or `min(dim(x))`).
#' @param nv Number of right singular vectors to return (0 or `min(dim(x))`).
#' @param ... Additional arguments (unused).
#' @return A list with components `d`, `u`, and `v`.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.svd}
#' @export
#' @method svd mlx
#' @examples
#' x <- as_mlx(matrix(c(1, 0, 0, 2), 2, 2))
#' svd(x)
svd.mlx <- function(x, nu = min(n, p), nv = min(n, p), ...) {
  if (!is.mlx(x)) x <- as_mlx(x)
  stopifnot(length(x$dim) == 2L)

  n <- x$dim[1]
  p <- x$dim[2]
  full_cols <- min(n, p)

  if (!nu %in% c(0L, full_cols)) {
    stop("svd.mlx only supports nu = 0 or nu = min(nrow, ncol).", call. = FALSE)
  }
  if (!nv %in% c(0L, full_cols)) {
    stop("svd.mlx only supports nv = 0 or nv = min(nrow, ncol).", call. = FALSE)
  }

  compute_uv <- (nu > 0 || nv > 0)
  res <- cpp_mlx_svd(x$ptr, compute_uv, x$dtype, x$device)

  if (!compute_uv) {
    s_ptr <- res[[1L]]
    d <- .mlx_wrap_result(s_ptr, x$device)
    return(list(d = d, u = NULL, v = NULL))
  }

  U <- .mlx_wrap_result(res$U, x$device)
  S <- .mlx_wrap_result(res$S, x$device)
  Vh <- .mlx_wrap_result(res$Vh, x$device)

  d <- S
  V <- .mlx_wrap_result(cpp_mlx_transpose(Vh$ptr), Vh$device)

  list(d = d, u = U, v = V)
}

#' Moore-Penrose pseudoinverse for MLX arrays
#'
#' @param x An `mlx` object or coercible matrix.
#' @return An `mlx` object containing the pseudoinverse.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.pinv}
#' @export
#' @examples
#' x <- as_mlx(matrix(c(1, 2, 3, 4), 2, 2))
#' pinv(x)
pinv <- function(x) {
  if (!is.mlx(x)) x <- as_mlx(x)
  stopifnot(length(x$dim) == 2L)

  ptr <- cpp_mlx_pinv(x$ptr, x$dtype, x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' Fast Fourier Transform
#'
#' Extends [stats::fft()] to work with `mlx` objects while delegating to the
#' standard R implementation for other inputs.
#'
#' @param z Input to transform. May be a numeric, complex, or `mlx` object.
#' @param inverse Logical flag; if `TRUE` compute the inverse transform.
#' @param ... Passed through to the default method.
#' @return For `mlx` inputs, an `mlx` object containing complex frequency
#'   coefficients; otherwise the base R result.
#' @seealso [stats::fft()], \url{https://ml-explore.github.io/mlx/build/html/python/fft.html#mlx.core.fft.fft}
#' @export
#' @examples
#' z <- as_mlx(c(1, 2, 3, 4))
#' fft(z)
#' fft(z, inverse = TRUE)
fft <- function(z, inverse = FALSE, ...) {
  UseMethod("fft")
}

#' @export
fft.default <- function(z, inverse = FALSE, ...) {
  stats::fft(z, inverse = inverse, ...)
}

#' @export
fft.mlx <- function(z, inverse = FALSE, ...) {
  if (!is.mlx(z)) z <- as_mlx(z)
  ptr <- cpp_mlx_fft(z$ptr, isTRUE(inverse), z$device)
  out <- .mlx_wrap_result(ptr, z$device)
  if (isTRUE(inverse)) {
    size <- prod(z$dim)
    out <- out * size
  }
  out
}

#' Matrix and vector norms for MLX tensors
#'
#' @param x An `mlx` array.
#' @param ord Numeric or character norm order. Use `NULL` for the default 2-norm.
#' @param axis Optional integer vector of axes (1-indexed) along which to compute the norm.
#' @param keepdims Logical; retain reduced axes with length one.
#' @return An `mlx` tensor containing the requested norm.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.norm}
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_norm(x)
#' mlx_norm(x, ord = 2)
#' mlx_norm(x, axis = 2)
mlx_norm <- function(x, ord = NULL, axis = NULL, keepdims = FALSE) {
  x <- if (is.mlx(x)) x else as_mlx(x)
  if (!is.null(ord) && !is.numeric(ord) && !is.character(ord)) {
    stop("ord must be numeric, character, or NULL.", call. = FALSE)
  }
  if (is.character(ord) && length(ord) == 1L) {
    ord <- toupper(ord)
  }
  axes_arg <- if (is.null(axis)) NULL else as.integer(axis)
  ptr <- cpp_mlx_norm(x$ptr, ord, axes_arg, keepdims, x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' Eigen decomposition for MLX tensors
#'
#' @param x An `mlx` square matrix.
#' @return A list with components `values` and `vectors`, both `mlx` tensors.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eig}
#' @export
#' @examples
#' x <- as_mlx(matrix(c(2, -1, 0, 2), 2, 2))
#' eig <- mlx_eig(x)
#' eig$values
#' eig$vectors
mlx_eig <- function(x) {
  if (!is.mlx(x)) x <- as_mlx(x)
  stopifnot(length(x$dim) == 2L, x$dim[1] == x$dim[2])

  res <- cpp_mlx_eig(x$ptr, x$device)
  list(
    values = .mlx_wrap_result(res$values, x$device),
    vectors = .mlx_wrap_result(res$vectors, x$device)
  )
}

#' Eigenvalues of MLX tensors
#'
#' @inheritParams mlx_eig
#' @return An `mlx` tensor containing eigenvalues.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigvals}
#' @export
#' @examples
#' x <- as_mlx(matrix(c(3, 1, 0, 2), 2, 2))
#' mlx_eigvals(x)
mlx_eigvals <- function(x) {
  if (!is.mlx(x)) x <- as_mlx(x)
  stopifnot(length(x$dim) == 2L, x$dim[1] == x$dim[2])
  ptr <- cpp_mlx_eigvals(x$ptr, x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' Eigenvalues of Hermitian MLX tensors
#'
#' @inheritParams mlx_eig
#' @param uplo Character string indicating which triangle to use ("L" or "U").
#' @return An `mlx` tensor containing eigenvalues.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigvalsh}
#' @export
#' @examples
#' x <- as_mlx(matrix(c(2, 1, 1, 3), 2, 2))
#' mlx_eigvalsh(x)
mlx_eigvalsh <- function(x, uplo = c("L", "U")) {
  if (!is.mlx(x)) x <- as_mlx(x)
  stopifnot(length(x$dim) == 2L, x$dim[1] == x$dim[2])
  uplo <- match.arg(uplo)
  ptr <- cpp_mlx_eigvalsh(x$ptr, uplo, x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' Eigen decomposition of Hermitian MLX tensors
#'
#' @inheritParams mlx_eigvalsh
#' @return A list with components `values` and `vectors`.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigh}
#' @export
#' @examples
#' x <- as_mlx(matrix(c(2, 1, 1, 3), 2, 2))
#' mlx_eigh(x)
mlx_eigh <- function(x, uplo = c("L", "U")) {
  if (!is.mlx(x)) x <- as_mlx(x)
  stopifnot(length(x$dim) == 2L, x$dim[1] == x$dim[2])
  uplo <- match.arg(uplo)
  res <- cpp_mlx_eigh(x$ptr, uplo, x$device)
  list(
    values = .mlx_wrap_result(res$values, x$device),
    vectors = .mlx_wrap_result(res$vectors, x$device)
  )
}

#' Solve triangular systems with MLX tensors
#'
#' @param a An `mlx` triangular matrix.
#' @param b Right-hand side matrix or vector.
#' @param upper Logical; if `TRUE`, `a` is upper triangular, otherwise lower.
#' @return An `mlx` tensor solution.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.solve_triangular}
#' @export
#' @examples
#' a <- as_mlx(matrix(c(2, 1, 0, 3), 2, 2))
#' b <- as_mlx(matrix(c(1, 5), 2, 1))
#' mlx_solve_triangular(a, b, upper = FALSE)
mlx_solve_triangular <- function(a, b, upper = FALSE) {
  if (!is.mlx(a)) a <- as_mlx(a)
  if (!is.mlx(b)) b <- as_mlx(b)
  stopifnot(length(a$dim) == 2L, a$dim[1] == a$dim[2])
  ptr <- cpp_mlx_solve_triangular(a$ptr, b$ptr, upper, a$device)
  .mlx_wrap_result(ptr, a$device)
}

#' Vector cross product with MLX tensors
#'
#' @param a,b Input `mlx` tensors containing 3D vectors.
#' @param axis Axis along which to compute the cross product (1-indexed, default last).
#' @return An `mlx` tensor of cross products.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.cross}
#' @export
#' @examples
#' u <- as_mlx(c(1, 0, 0))
#' v <- as_mlx(c(0, 1, 0))
#' mlx_cross(u, v)
mlx_cross <- function(a, b, axis = -1L) {
  if (!is.mlx(a)) a <- as_mlx(a)
  if (!is.mlx(b)) b <- as_mlx(b)
  ptr <- cpp_mlx_cross(a$ptr, b$ptr, as.integer(axis), a$device)
  .mlx_wrap_result(ptr, a$device)
}
