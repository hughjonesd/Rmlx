#' @keywords internal
.mlx_wrap_result <- function(ptr, device) {
  dim <- cpp_mlx_shape(ptr)
  dtype <- cpp_mlx_dtype(ptr)
  new_mlx(ptr, dim, dtype, device)
}

#' @export
#' @method chol mlx
chol.mlx <- function(x, pivot = FALSE, LINPACK = FALSE, ...) {
  if (!is.mlx(x)) x <- as_mlx(x)
  if (pivot) stop("pivoted Cholesky is not supported for mlx objects.", call. = FALSE)
  if (LINPACK) stop("LINPACK routines are not supported for mlx objects.", call. = FALSE)

  ptr <- cpp_mlx_cholesky(x$ptr, TRUE, x$dtype, x$device)
  .mlx_wrap_result(ptr, x$device)
}

#' @export
#' @method qr mlx
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
#' @method svd mlx
svd.mlx <- function(x, nu = min(n, p), nv = min(n, p), LINPACK = FALSE, ...) {
  if (!is.mlx(x)) x <- as_mlx(x)
  stopifnot(length(x$dim) == 2L)

  if (LINPACK) stop("LINPACK routines are not supported for mlx objects.", call. = FALSE)

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
    d <- as.numeric(as.matrix(.mlx_wrap_result(s_ptr, x$device)))
    return(list(d = d, u = NULL, v = NULL))
  }

  U <- .mlx_wrap_result(res$U, x$device)
  S <- .mlx_wrap_result(res$S, x$device)
  Vh <- .mlx_wrap_result(res$Vh, x$device)

  d <- as.numeric(as.matrix(S))
  V <- .mlx_wrap_result(cpp_mlx_transpose(Vh$ptr), Vh$device)

  list(d = d, u = U, v = V)
}

#' Moore-Penrose pseudoinverse for MLX arrays
#'
#' @param x An `mlx` object or coercible matrix.
#' @return An `mlx` object containing the pseudoinverse.
#' @export
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
#' @seealso [stats::fft()]
#' @export
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
