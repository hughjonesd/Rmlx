#' Cholesky decomposition for mlx arrays
#'
#' If `x` is not symmetric positive semi-definite, "behaviour is undefined"
#' according to the MLX documentation.
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_matrix_required
#' @param pivot Ignored; pivoted decomposition is not supported.
#' @inheritParams ellipsis_ignored
#' @return Upper-triangular Cholesky factor as an mlx matrix.
#' @seealso [mlx.linalg.cholesky](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.cholesky)
#' @export
#' @examples
#' x <- mlx_matrix(c(4, 1, 1, 3), 2, 2, device = "cpu")
#' chol(x)
chol.mlx <- function(x, pivot = FALSE, ...) {
  x <- as_mlx(x)
  if (pivot) stop("pivoted Cholesky is not supported for mlx objects.", call. = FALSE)
  x_dtype <- mlx_dtype(x)
  ptr <- cpp_mlx_cholesky(x$ptr, TRUE, x_dtype, mlx_device(x))
  new_mlx(ptr, mlx_device(x))
}

#' Inverse from Cholesky decomposition
#'
#' Compute the inverse of a symmetric, positive definite matrix from its
#' Cholesky decomposition. The input `x` should be an upper triangular matrix
#' from `chol()`.
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_matrix_required
#' @param size Ignored; included for compatibility with base R.
#' @inheritParams ellipsis_ignored
#' @return The inverse of the original matrix (before Cholesky decomposition).
#' @seealso [chol()], [solve()], [mlx_cholesky_inv()]
#' @export
#' @examples
#' A <- mlx_matrix(c(4, 1, 1, 3), 2, 2, device = "cpu")
#' U <- chol(A)
#' A_inv <- chol2inv(U)
#' # Verify: A %*% A_inv should be identity
#' A %*% A_inv
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
  x <- as_mlx(x)
  # R's chol() always returns upper triangular, so we always use upper=TRUE
  mlx_cholesky_inv(x, upper = TRUE)
}

#' QR decomposition for mlx arrays
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_matrix_required
#' @param tol Ignored; custom tolerances are not supported.
#' @param LAPACK Ignored; set to `FALSE`.
#' @inheritParams ellipsis_ignored
#' @return A list with components `Q` and `R`, each an mlx matrix.
#' @seealso [mlx.linalg.qr](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.qr)
#' @export
#' @examples
#' x <- mlx_matrix(c(1, 2, 3, 4, 5, 6), 3, 2, device = "cpu")
#' qr(x)
qr.mlx <- function(x, tol = 1e-7, LAPACK = FALSE, ...) {
  x <- as_mlx(x)
  stopifnot(length(dim(x)) == 2L)
  x_dtype <- mlx_dtype(x)

  if (!missing(tol) && !isTRUE(all.equal(tol, 1e-7))) {
    stop("Custom tolerance is not supported for mlx QR decomposition.", call. = FALSE)
  }
  if (LAPACK) stop("LAPACK = TRUE is not supported for mlx objects.", call. = FALSE)

  res <- cpp_mlx_qr(x$ptr, x_dtype, mlx_device(x))
  device <- mlx_device(x)
  structure(
    list(
      Q = new_mlx(res$Q, device),
      R = new_mlx(res$R, device)
    ),
    class = c("mlx_qr", "list")
  )
}

#' Singular value decomposition
#'
#' Generic function for SVD computation.
#' @param x An object.
#' @inheritParams ellipsis_base
#' @return A list with components `d`, `u`, and `v`.
#' @export
svd <- function(x, ...) {
  UseMethod("svd")
}

#' @export
svd.default <- function(x, ...) base::svd(x, ...)

#' Singular value decomposition for mlx arrays
#'
#' Note that mlx's svd returns "full" SVD, with U and V' both square matrices.
#' This is different from R's implementation.
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_matrix_required
#' @param nu Number of left singular vectors to return (0 or `min(dim(x))`).
#' @param nv Number of right singular vectors to return (0 or `min(dim(x))`).
#' @inheritParams ellipsis_ignored
#' @return A list with components `d`, `u`, and `v`.
#' @seealso [mlx.linalg.svd](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.svd)
#' @export
#' @examples
#' x <- mlx_matrix(c(1, 0, 0, 2), 2, 2, device = "cpu")
#' svd(x)
svd.mlx <- function(x, nu = min(n, p), nv = min(n, p), ...) {
  x <- as_mlx(x)
  stopifnot(length(dim(x)) == 2L)

  n <- dim(x)[1]
  p <- dim(x)[2]
  full_cols <- min(n, p)

  if (!nu %in% c(0L, full_cols)) {
    stop("svd.mlx only supports nu = 0 or nu = min(nrow, ncol).", call. = FALSE)
  }
  if (!nv %in% c(0L, full_cols)) {
    stop("svd.mlx only supports nv = 0 or nv = min(nrow, ncol).", call. = FALSE)
  }

  compute_uv <- (nu > 0 || nv > 0)
  x_dtype <- mlx_dtype(x)
  res <- cpp_mlx_svd(x$ptr, compute_uv, x_dtype, mlx_device(x))

  if (!compute_uv) {
    s_ptr <- res[[1L]]
    d <- new_mlx(s_ptr, mlx_device(x))
    return(list(d = d, u = NULL, v = NULL))
  }

  U <- new_mlx(res$U, mlx_device(x))
  S <- new_mlx(res$S, mlx_device(x))
  Vh <- new_mlx(res$Vh, mlx_device(x))

  d <- S
  V <- new_mlx(cpp_mlx_transpose(Vh$ptr), mlx_device(Vh))

  list(d = d, u = U, v = V)
}

#' Moore-Penrose pseudoinverse for MLX arrays
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @param x An mlx object or coercible matrix.
#' @return An mlx object containing the pseudoinverse.
#' @seealso [mlx.linalg.pinv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.pinv)
#' @export
#' @examples
#' x <- mlx_matrix(c(1, 2, 3, 4), 2, 2, device = "cpu")
#' pinv(x)
pinv <- function(x) {
  x <- as_mlx(x)
  stopifnot(length(dim(x)) == 2L)

  x_dtype <- mlx_dtype(x)
  ptr <- cpp_mlx_pinv(x$ptr, x_dtype, mlx_device(x))
  new_mlx(ptr, mlx_device(x))
}

#' Fast Fourier Transform
#'
#' Extends [stats::fft()] to work with mlx objects while delegating to the
#' standard R implementation for other inputs.
#'
#' @param z Input to transform. May be a numeric, complex, or mlx object.
#' @param inverse Logical flag; if `TRUE` compute the inverse transform.
#' @inheritParams common_params
#' @inheritParams ellipsis_base
#' @return For mlx inputs, an mlx object containing complex frequency
#'   coefficients; otherwise the base R result.
#' @seealso [stats::fft()], [mlx_fft()], [mlx_fft2()], [mlx_fftn()], [mlx.core.fft.fft](https://ml-explore.github.io/mlx/build/html/python/fft.html#mlx.core.fft.fft)
#' @export
#' @examples
#' z <- as_mlx(c(1, 2, 3, 4))
#' fft(z)
#' fft(z, inverse = TRUE)
fft <- function(z, inverse = FALSE, ...) {
  UseMethod("fft")
}

#' @export
#' @rdname fft
fft.default <- function(z, inverse = FALSE, ...) {
  stats::fft(z, inverse = inverse, ...)
}

#' @export
#' @rdname fft
fft.mlx <- function(z, inverse = FALSE, axis, ...) {
  mlx_fft(z, axis = axis, inverse = inverse)
}

#' Matrix and vector norms for mlx arrays
#'
#' @inheritParams mlx_array_required
#' @param ord Numeric or character norm order. Use `NULL` for the default 2-norm.
#' @inheritParams common_params
#' @return An mlx array containing the requested norm.
#' @seealso [mlx.linalg.norm](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.norm)
#' @export
#' @examples
#' x <- mlx_matrix(1:4, 2, 2)
#' mlx_norm(x)
#' mlx_norm(x, ord = 2)
#' mlx_norm(x, axes = 2)
mlx_norm <- function(x, ord = NULL, axes = NULL, drop = TRUE) {
  x <- as_mlx(x)
  if (!is.null(ord) && !is.numeric(ord) && !is.character(ord)) {
    stop("ord must be numeric, character, or NULL.", call. = FALSE)
  }
  if (is.character(ord) && length(ord) == 1L) {
    ord <- toupper(ord)
  }
  axes_arg <- if (is.null(axes)) NULL else as.integer(axes)
  ptr <- cpp_mlx_norm(x$ptr, ord, axes_arg, !isTRUE(drop), mlx_device(x))
  new_mlx(ptr, mlx_device(x))
}

#' Eigen decomposition for mlx arrays
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_matrix_required
#' @return A list with components `values` and `vectors`, both mlx arrays.
#' @seealso [mlx.linalg.eig](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eig)
#' @export
#' @examples
#' x <- mlx_matrix(c(2, -1, 0, 2), 2, 2, device = "cpu")
#' eig <- mlx_eig(x)
#' eig$values
#' eig$vectors
mlx_eig <- function(x) {
  x <- as_mlx(x)
  stopifnot(length(dim(x)) == 2L, dim(x)[1] == dim(x)[2])

  res <- cpp_mlx_eig(x$ptr, mlx_device(x))
  list(
    values = new_mlx(res$values, mlx_device(x)),
    vectors = new_mlx(res$vectors, mlx_device(x))
  )
}

#' Eigenvalues of mlx arrays
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_eig
#' @return An mlx array containing eigenvalues.
#' @seealso [mlx.linalg.eigvals](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigvals)
#' @export
#' @examples
#' x <- mlx_matrix(c(3, 1, 0, 2), 2, 2, device = "cpu")
#' mlx_eigvals(x)
mlx_eigvals <- function(x) {
  x <- as_mlx(x)
  stopifnot(length(dim(x)) == 2L, dim(x)[1] == dim(x)[2])
  ptr <- cpp_mlx_eigvals(x$ptr, mlx_device(x))
  new_mlx(ptr, mlx_device(x))
}

#' Eigenvalues of Hermitian mlx arrays
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_eig
#' @param uplo Character string indicating which triangle to use ("L" or "U").
#' @return An mlx array containing eigenvalues.
#' @seealso [mlx.linalg.eigvalsh](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigvalsh)
#' @export
#' @examples
#' x <- mlx_matrix(c(2, 1, 1, 3), 2, 2, device = "cpu")
#' mlx_eigvalsh(x)
mlx_eigvalsh <- function(x, uplo = c("L", "U")) {
  x <- as_mlx(x)
  stopifnot(length(dim(x)) == 2L, dim(x)[1] == dim(x)[2])
  uplo <- match.arg(uplo)
  ptr <- cpp_mlx_eigvalsh(x$ptr, uplo, mlx_device(x))
  new_mlx(ptr, mlx_device(x))
}

#' Eigen decomposition of Hermitian mlx arrays
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_eigvalsh
#' @return A list with components `values` and `vectors`.
#' @seealso [mlx.linalg.eigh](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigh)
#' @export
#' @examples
#' x <- mlx_matrix(c(2, 1, 1, 3), 2, 2, device = "cpu")
#' mlx_eigh(x)
mlx_eigh <- function(x, uplo = c("L", "U")) {
  x <- as_mlx(x)
  stopifnot(length(dim(x)) == 2L, dim(x)[1] == dim(x)[2])
  uplo <- match.arg(uplo)
  res <- cpp_mlx_eigh(x$ptr, uplo, mlx_device(x))
  list(
    values = new_mlx(res$values, mlx_device(x)),
    vectors = new_mlx(res$vectors, mlx_device(x))
  )
}

#' Solve triangular systems with mlx arrays
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @param a An mlx triangular matrix.
#' @param b Right-hand side matrix or vector.
#' @param upper Logical; if `TRUE`, `a` is upper triangular, otherwise lower.
#' @inheritParams ellipsis_base
#' @return An mlx array solution.
#' @seealso [mlx.linalg.solve_triangular](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.solve_triangular)
#' @export
#' @examples
#' a <- mlx_matrix(c(2, 1, 0, 3), 2, 2, device = "cpu")
#' b <- mlx_matrix(c(1, 5), 2, 1, device = "cpu")
#' mlx_solve_triangular(a, b, upper = FALSE)
mlx_solve_triangular <- function(a, b, upper = FALSE) {
  a <- as_mlx(a)
  b <- as_mlx(b)
  stopifnot(length(dim(a)) == 2L, dim(a)[1] == dim(a)[2])
  ptr <- cpp_mlx_solve_triangular(a$ptr, b$ptr, upper, mlx_device(a))
  new_mlx(ptr, mlx_device(a))
}

#' @rdname mlx_solve_triangular
#' @param r Triangular system matrix passed to [backsolve()].
#' @param x Right-hand side supplied to [backsolve()].
#' @param k Number of columns of `r` to use.
#' @param upper.tri Logical; indicates if `r` is upper triangular.
#' @param transpose Logical; if `TRUE`, solve `t(r) %*% x = b`.
#' @export
backsolve <- function(r, x, k = NULL, upper.tri = TRUE, transpose = FALSE, ...) {
  UseMethod("backsolve")
}

#' @rdname mlx_solve_triangular
#' @export
backsolve.default <- function(r, x, k = NULL, upper.tri = TRUE, transpose = FALSE, ...) {
  base::backsolve(r, x, k = if (is.null(k)) ncol(r) else k, upper.tri = upper.tri, transpose = transpose, ...)
}

#' @rdname mlx_solve_triangular
#' @export
backsolve.mlx <- function(r, x, k = NULL, upper.tri = TRUE, transpose = FALSE, ...) {
  r_mlx <- as_mlx(r)
  x_mlx <- as_mlx(x, device = mlx_device(r_mlx))

  if (length(dim(r_mlx)) != 2L) {
    stop("`r` must be a matrix when using backsolve() with mlx arrays.", call. = FALSE)
  }

  if (is.null(k)) {
    k <- dim(r_mlx)[2L]
  }
  if (!identical(k, dim(r_mlx)[2L])) {
    stop("`k` values other than ncol(r) are not yet supported for mlx arrays.", call. = FALSE)
  }

  target <- r_mlx
  if (transpose) {
    target <- t(target)
    upper.tri <- !upper.tri
  }

  mlx_solve_triangular(target, x_mlx, upper = upper.tri)
}

#' Vector cross product with mlx arrays
#'
#' @param a,b Input mlx arrays containing 3D vectors.
#' @param axis Axis along which to compute the cross product (1-indexed).
#'   Omit the argument to use the trailing dimension.
#' @return An mlx array of cross products.
#' @seealso [mlx.linalg.cross](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.cross)
#' @export
#' @examples
#' u <- as_mlx(c(1, 0, 0))
#' v <- as_mlx(c(0, 1, 0))
#' mlx_cross(u, v)
mlx_cross <- function(a, b, axis = NULL) {
  a <- as_mlx(a)
  b <- as_mlx(b)
  shape_len <- length(mlx_shape(a))
  axis_val <- if (missing(axis) || is.null(axis)) {
    if (shape_len == 0L) {
      stop("`axis` must be supplied for scalar inputs.", call. = FALSE)
    }
    shape_len
  } else axis
  if (length(axis_val) != 1L || is.na(axis_val) || axis_val < 1L) {
    stop("`axis` must be NULL or a single positive integer.", call. = FALSE)
  }
  ptr <- cpp_mlx_cross(a$ptr, b$ptr, as.integer(axis_val))
  new_mlx(ptr, mlx_device(a))
}

#' Matrix trace for mlx arrays
#'
#' Computes the sum of the diagonal elements of a 2D array, or the sum along
#' diagonals of a higher dimensional array.
#'
#' @inheritParams mlx_array_required
#' @param offset Offset of the diagonal (0 for main diagonal, positive for above, negative for below).
#' @param axis1,axis2 Axes along which the diagonals are taken (1-indexed, default 1 and 2).
#' @return An mlx scalar or array containing the trace.
#' @seealso [mlx.core.trace](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.trace.html)
#' @export
#' @examples
#' x <- mlx_matrix(1:9, 3, 3)
#' mlx_trace(x)
#' mlx_trace(x, offset = 1)
mlx_trace <- function(x, offset = 0L, axis1 = 1L, axis2 = 2L) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_trace(x$ptr, as.integer(offset), as.integer(axis1), as.integer(axis2))
  new_mlx(ptr, mlx_device(x))
}

#' Extract diagonal or construct diagonal matrix for mlx arrays
#'
#' Extract a diagonal from a matrix or construct a diagonal matrix from a vector.
#'
#' @param x An mlx array. If 1D, creates a diagonal matrix. If 2D or higher, extracts the diagonal.
#' @param offset Diagonal offset (0 for main diagonal, positive for above, negative for below).
#' @param axis1,axis2 For multi-dimensional arrays, which axes define the 2D planes (1-indexed).
#' @return An mlx array.
#' @seealso [mlx.core.diagonal](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.diagonal.html)
#' @export
#' @examples
#' # Extract diagonal
#' x <- mlx_matrix(1:9, 3, 3)
#' mlx_diagonal(x)
#' # (Constructing diagonals from 1D inputs is not yet supported.)
mlx_diagonal <- function(x, offset = 0L, axis1 = 1L, axis2 = 2L) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_diagonal(x$ptr, as.integer(offset), as.integer(axis1), as.integer(axis2))
  new_mlx(ptr, mlx_device(x))
}

#' Outer product of two vectors
#'
#' @param X,Y Numeric vectors or mlx arrays.
#' @param FUN Function to apply (for default method).
#' @inheritParams ellipsis_base
#' @return For mlx inputs, an mlx matrix. Otherwise delegates to `base::outer`.
#' @seealso [mlx.core.outer](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.outer.html)
#' @export
#' @examples
#' x <- as_mlx(c(1, 2, 3))
#' y <- as_mlx(c(4, 5))
#' outer(x, y)
outer <- function(X, Y, FUN = "*", ...) {
  UseMethod("outer")
}

#' @export
outer.default <- base::outer

#' @export
#' @rdname outer
outer.mlx <- function(X, Y, FUN = "*", ...) {
  X <- as_mlx(X)
  Y <- as_mlx(Y)
  ptr <- cpp_mlx_outer(X$ptr, Y$ptr)
  new_mlx(ptr, mlx_device(X))
}

#' Unflatten an axis into multiple axes
#'
#' The reverse of flattening: expands a single axis into multiple axes with the given shape.
#'
#' @inheritParams mlx_array_required
#' @param axis Which axis to unflatten (1-indexed).
#' @param shape Integer vector specifying the new shape for the unflattened axis.
#' @return An mlx array with the axis expanded.
#' @seealso [mlx.core.unflatten](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.unflatten.html)
#' @export
#' @examples
#' # Flatten and unflatten
#' x <- mlx_array(1:24, c(2, 3, 4))
#' x_flat <- mlx_reshape(x, c(2, 12))  # flatten last two dims
#' mlx_unflatten(x_flat, axis = 2, shape = c(3, 4))  # restore original shape
mlx_unflatten <- function(x, axis, shape) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_unflatten(x$ptr, as.integer(axis), as.integer(shape))
  new_mlx(ptr, mlx_device(x))
}

#' Compute matrix inverse
#'
#' Computes the inverse of a square matrix. Note that as of MLX 0.30.0, this
#' runs on the CPU.
#'
#' @inheritParams mlx_array_required
#' @inherit mlx_cpu_only_operation details
#' @return The inverse of `x`.
#' @seealso [mlx.core.linalg.inv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.inv)
#' @export
#' @examples
#' A <- mlx_matrix(c(4, 7, 2, 6), 2, 2, device = "cpu")
#' A_inv <- mlx_inv(A)
#' # Verify: A %*% A_inv should be identity
#' A %*% A_inv
mlx_inv <- function(x) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_inv(x$ptr, mlx_device(x))
  new_mlx(ptr, mlx_device(x))
}

#' Compute triangular matrix inverse
#'
#' Computes the inverse of a triangular matrix.
#'
#' **Note:** MLX may crash if `x` is not triangular.
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_array_required
#' @param upper Logical; if `TRUE`, `x` is upper triangular, otherwise lower triangular.
#' @return The inverse of the triangular matrix `x`.
#' @seealso [mlx.core.linalg.tri_inv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.tri_inv)
#' @export
#' @examples
#' # Lower triangular matrix
#' L <- mlx_matrix(c(1:3, 0, 4:5, 0, 0, 6), 3, 3, device = "cpu")
#' mlx_tri_inv(L, upper = FALSE)
mlx_tri_inv <- function(x, upper = FALSE) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_tri_inv(x$ptr, upper, mlx_device(x))
  new_mlx(ptr, mlx_device(x))
}

#' Compute matrix inverse via Cholesky decomposition
#'
#' Computes the inverse of a positive definite matrix from its Cholesky factor.
#' Note: `x` should be the Cholesky factor (L or U), not the original matrix.
#'
#' For a more R-like interface, see [chol2inv()].
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_array_required
#' @param upper Logical; if `TRUE`, `x` is upper triangular, otherwise lower triangular.
#' @return The inverse of the original matrix (A^-1 where A = LL' or A = U'U).
#' @seealso [chol2inv()], [mlx.core.linalg.cholesky_inv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.cholesky_inv)
#' @export
#' @examples
#' # Create a positive definite matrix
#' A <- matrix(rnorm(9), 3, 3)
#' A <- t(A) %*% A
#' # Compute Cholesky factor
#' L <- chol(A, pivot = FALSE, upper = FALSE)
#' # Get inverse from Cholesky factor
#' mlx_cholesky_inv(as_mlx(L, device = "cpu"))
mlx_cholesky_inv <- function(x, upper = FALSE) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_cholesky_inv(x$ptr, upper, mlx_device(x))
  new_mlx(ptr, mlx_device(x))
}

#' LU factorization
#'
#' Computes the LU factorization of a matrix.
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_array_required
#' @return A list with components `p` (pivot indices), `l` (lower triangular),
#'   and `u` (upper triangular). The relationship is `A = L[P, ] %*% U`.
#' @seealso [mlx.core.linalg.lu](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.lu)
#' @export
#' @examples
#' A <- mlx_matrix(rnorm(16), 4, 4, device = "cpu")
#' lu_result <- mlx_lu(A)
#' P <- lu_result$p  # Pivot indices
#' L <- lu_result$l  # Lower triangular
#' U <- lu_result$u  # Upper triangular
mlx_lu <- function(x) {
  x <- as_mlx(x)
  result <- cpp_mlx_lu(x$ptr, mlx_device(x))
  list(
    p = new_mlx(result$p, mlx_device(x)),
    l = new_mlx(result$l, mlx_device(x)),
    u = new_mlx(result$u, mlx_device(x))
  )
}
