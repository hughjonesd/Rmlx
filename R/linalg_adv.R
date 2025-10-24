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

#' Inverse from Cholesky decomposition
#'
#' Compute the inverse of a symmetric, positive definite matrix from its
#' Cholesky decomposition.
#'
#' @param x An `mlx` matrix containing the Cholesky decomposition (upper triangular).
#'   This should be the result of `chol()`.
#' @param size Ignored; included for compatibility with base R.
#' @param ... Additional arguments (unused).
#' @return The inverse of the original matrix (before Cholesky decomposition).
#' @seealso [chol()], [solve()], [mlx_cholesky_inv()]
#' @export
#' @examples
#' A <- as_mlx(matrix(c(4, 1, 1, 3), 2, 2))
#' U <- chol(A)
#' A_inv <- chol2inv(U)
#' # Verify: A %*% A_inv should be identity
#' as.matrix(A %*% A_inv)
chol2inv <- function(x, size = NCOL(x), ...) {
  UseMethod("chol2inv")
}

#' @export
#' @rdname chol2inv
chol2inv.default <- function(x, size = NCOL(x), ...) {
  # Call base R's chol2inv
  base::chol2inv(x, size = size, ...)
}

#' @export
#' @rdname chol2inv
chol2inv.mlx <- function(x, size = NCOL(x), ...) {
  if (!is.mlx(x)) x <- as_mlx(x)
  # R's chol() always returns upper triangular, so we always use upper=TRUE
  mlx_cholesky_inv(x, upper = TRUE)
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

#' Matrix trace for MLX tensors
#'
#' Computes the sum of the diagonal elements of a 2D array, or the sum along
#' diagonals of a higher dimensional array.
#'
#' @param x An `mlx` array.
#' @param offset Offset of the diagonal (0 for main diagonal, positive for above, negative for below).
#' @param axis1,axis2 Axes along which the diagonals are taken (1-indexed, default 1 and 2).
#' @return An `mlx` scalar or array containing the trace.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.trace.html}
#' @export
#' @examples
#' x <- as_mlx(matrix(1:9, 3, 3))
#' mlx_trace(x)
#' mlx_trace(x, offset = 1)
mlx_trace <- function(x, offset = 0L, axis1 = 1L, axis2 = 2L) {
  if (!is.mlx(x)) x <- as_mlx(x)
  ptr <- cpp_mlx_trace(x$ptr, as.integer(offset), as.integer(axis1), as.integer(axis2), x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' Extract diagonal or construct diagonal matrix for MLX tensors
#'
#' Extract a diagonal from a matrix or construct a diagonal matrix from a vector.
#'
#' @param x An `mlx` array. If 1D, creates a diagonal matrix. If 2D or higher, extracts the diagonal.
#' @param offset Diagonal offset (0 for main diagonal, positive for above, negative for below).
#' @param axis1,axis2 For multi-dimensional arrays, which axes define the 2D planes (1-indexed).
#' @return An `mlx` tensor.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.diagonal.html}
#' @export
#' @examples
#' # Extract diagonal
#' x <- as_mlx(matrix(1:9, 3, 3))
#' mlx_diagonal(x)
#'
#' # Create diagonal matrix
#' v <- as_mlx(c(1, 2, 3))
#' mlx_diagonal(v)
mlx_diagonal <- function(x, offset = 0L, axis1 = 1L, axis2 = 2L) {
  if (!is.mlx(x)) x <- as_mlx(x)
  ptr <- cpp_mlx_diagonal(x$ptr, as.integer(offset), as.integer(axis1), as.integer(axis2), x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' Outer product of two vectors
#'
#' @param x,y Numeric vectors or `mlx` arrays.
#' @param ... Additional arguments passed to methods.
#' @return For `mlx` inputs, an `mlx` matrix. Otherwise delegates to `base::outer`.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.outer.html}
#' @export
#' @examples
#' x <- as_mlx(c(1, 2, 3))
#' y <- as_mlx(c(4, 5))
#' outer(x, y)
outer <- function(x, y, ...) {
  UseMethod("outer")
}

#' @export
outer.default <- base::outer

#' @export
#' @rdname outer
outer.mlx <- function(x, y, ...) {
  if (!is.mlx(x)) x <- as_mlx(x)
  if (!is.mlx(y)) y <- as_mlx(y)
  ptr <- cpp_mlx_outer(x$ptr, y$ptr, x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' Unflatten an axis into multiple axes
#'
#' The reverse of flattening: expands a single axis into multiple axes with the given shape.
#'
#' @param x An `mlx` array.
#' @param axis Which axis to unflatten (1-indexed).
#' @param shape Integer vector specifying the new shape for the unflattened axis.
#' @return An `mlx` array with the axis expanded.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.unflatten.html}
#' @export
#' @examples
#' # Flatten and unflatten
#' x <- as_mlx(array(1:24, c(2, 3, 4)))
#' x_flat <- mlx_reshape(x, c(2, 12))  # flatten last two dims
#' mlx_unflatten(x_flat, axis = 2, shape = c(3, 4))  # restore original shape
mlx_unflatten <- function(x, axis, shape) {
  if (!is.mlx(x)) x <- as_mlx(x)
  ptr <- cpp_mlx_unflatten(x$ptr, as.integer(axis), as.integer(shape), x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' Compute matrix inverse
#'
#' Computes the inverse of a square matrix.
#'
#' @param x An `mlx` array (must be square).
#' @return The inverse of `x`.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.inv}
#' @export
#' @examples
#' A <- as_mlx(matrix(c(4, 7, 2, 6), 2, 2))
#' A_inv <- mlx_inv(A)
#' # Verify: A %*% A_inv should be identity
#' as.matrix(A %*% A_inv)
mlx_inv <- function(x) {
  if (!is.mlx(x)) x <- as_mlx(x)
  ptr <- cpp_mlx_inv(x$ptr, x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' Compute triangular matrix inverse
#'
#' Computes the inverse of a triangular matrix.
#'
#' @param x An `mlx` array (triangular matrix).
#' @param upper Logical; if `TRUE`, `x` is upper triangular, otherwise lower triangular.
#' @return The inverse of the triangular matrix `x`.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.tri_inv}
#' @export
#' @examples
#' # Lower triangular matrix
#' L <- as_mlx(matrix(c(1, 2, 0, 3, 0, 0, 4, 5, 6), 3, 3, byrow = TRUE))
#' L_inv <- mlx_tri_inv(L, upper = FALSE)
mlx_tri_inv <- function(x, upper = FALSE) {
  if (!is.mlx(x)) x <- as_mlx(x)
  ptr <- cpp_mlx_tri_inv(x$ptr, upper, x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' Compute matrix inverse via Cholesky decomposition
#'
#' Computes the inverse of a positive definite matrix from its Cholesky factor.
#' Note: `x` should be the Cholesky factor (L or U), not the original matrix.
#'
#' For a more R-like interface, see [chol2inv()].
#'
#' @param x An `mlx` array containing the Cholesky factor (lower or upper triangular).
#' @param upper Logical; if `TRUE`, `x` is upper triangular, otherwise lower triangular.
#' @return The inverse of the original matrix (A^-1 where A = LL' or A = U'U).
#' @seealso [chol2inv()], \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.cholesky_inv}
#' @export
#' @examples
#' # Create a positive definite matrix
#' A <- matrix(rnorm(9), 3, 3)
#' A <- t(A) %*% A
#' # Compute Cholesky factor
#' L <- chol(A, pivot = FALSE, upper = FALSE)
#' # Get inverse from Cholesky factor
#' A_inv <- mlx_cholesky_inv(as_mlx(L))
mlx_cholesky_inv <- function(x, upper = FALSE) {
  if (!is.mlx(x)) x <- as_mlx(x)
  ptr <- cpp_mlx_cholesky_inv(x$ptr, upper, x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' LU factorization
#'
#' Computes the LU factorization of a matrix.
#'
#' @param x An `mlx` array.
#' @return A list with components `p` (pivot indices), `l` (lower triangular),
#'   and `u` (upper triangular). The relationship is A = L[P, :] \%*\% U.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.lu}
#' @export
#' @examples
#' A <- as_mlx(matrix(rnorm(16), 4, 4))
#' lu_result <- mlx_lu(A)
#' P <- lu_result$p  # Pivot indices
#' L <- lu_result$l  # Lower triangular
#' U <- lu_result$u  # Upper triangular
mlx_lu <- function(x) {
  if (!is.mlx(x)) x <- as_mlx(x)
  result <- cpp_mlx_lu(x$ptr, x$device)
  list(
    p = .mlx_wrap_result(result$p, x$device),
    l = .mlx_wrap_result(result$l, x$device),
    u = .mlx_wrap_result(result$u, x$device)
  )
}
