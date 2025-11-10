#' Math operations for MLX arrays
#'
#' @inheritParams mlx_array_required
#' @param ... Additional arguments (ignored)
#' @return An mlx object with the result
#' @seealso [mlx.core.array](https://ml-explore.github.io/mlx/build/html/python/array.html)
#' @aliases abs.mlx sign.mlx sqrt.mlx floor.mlx ceiling.mlx trunc.mlx round.mlx signif.mlx exp.mlx log.mlx log10.mlx log2.mlx log1p.mlx expm1.mlx cos.mlx sin.mlx tan.mlx acos.mlx asin.mlx atan.mlx cosh.mlx sinh.mlx tanh.mlx acosh.mlx asinh.mlx atanh.mlx cospi.mlx sinpi.mlx tanpi.mlx cumsum.mlx cumprod.mlx cummax.mlx cummin.mlx
#' @export
#' @method Math mlx
#' @examples
#' x <- as_mlx(matrix(c(-1, 0, 1), 3, 1))
#' sin(x)
#' round(x + 0.4)
Math.mlx <- function(x, ...) {
  # .Generic contains the name of the function that was called
  op <- .Generic
  dots <- list(...)

  # Cumulative operations flatten the array in column-major order (like R)
  # MLX flattens in row-major order, so we need to fall back to R
  if (op %in% c("cumsum", "cumprod", "cummax", "cummin")) {
    ptr <- cpp_mlx_cumulative(x$ptr, op)
    len <- as.integer(prod(x$dim))
    return(new_mlx(ptr, len, x$dtype, x$device))
  }

  # Map R function names to MLX operations
  # Most are direct matches, but some need mapping
  op_map <- c(
    "ceiling" = "ceil"
  )

  # Additional arguments change semantics for some Math generics (e.g., log base, round digits)
  if (length(dots) > 0 && op %in% c("log", "round", "signif")) {
    x_r <- as.matrix(x)
    result_r <- do.call(get(op, mode = "function"), c(list(x_r), dots))
    return(as_mlx(result_r, dtype = x$dtype, device = x$device))
  }

  if (op %in% names(op_map)) {
    op <- op_map[op]
  }

  # Try MLX operation first
  result <- tryCatch({
    ptr <- cpp_mlx_unary(x$ptr, op)
    new_mlx(ptr, x$dim, x$dtype, x$device)
  }, error = function(e) {
    # If MLX doesn't support this operation, fall back to base R
    if (grepl("Unsupported unary operation", e$message)) {
      warning("MLX does not support '", .Generic, "', falling back to R implementation",
              call. = FALSE)
      # Convert to R matrix, apply operation, convert back
      x_r <- as.matrix(x)
      result_r <- get(.Generic, mode = "function")(x_r, ...)
      as_mlx(result_r, dtype = x$dtype, device = x$device)
    } else {
      # Re-throw other errors
      stop(e)
    }
  })

  result
}

#' Element-wise approximate equality
#'
#' Returns a boolean array indicating which elements of two arrays are close
#' within specified tolerances.
#'
#' @param a,b MLX arrays or objects coercible to MLX arrays
#' @param rtol Relative tolerance (default: 1e-5)
#' @param atol Absolute tolerance (default: 1e-8)
#' @param equal_nan If `TRUE`, NaN values are considered equal (default: `FALSE`)
#' @inheritParams common_params
#'
#' @details
#' Two values are considered close if:
#' \code{abs(a - b) <= (atol + rtol * abs(b))}
#'
#' Infinite values with matching signs are considered equal.
#' Supports NumPy-style broadcasting.
#'
#' @return An mlx array of booleans with element-wise comparison results
#'
#' @seealso [mlx_allclose()], [all.equal.mlx()],
#'   [mlx.core.isclose](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.isclose.html)
#' @export
#' @examples
#' a <- as_mlx(c(1.0, 2.0, 3.0))
#' b <- as_mlx(c(1.0 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-3))
#' mlx_isclose(a, b)  # First two TRUE, last FALSE
mlx_isclose <- function(a, b, rtol = 1e-5, atol = 1e-8, equal_nan = FALSE,
                        device = mlx_default_device()) {
  a <- as_mlx(a)
  b <- as_mlx(b)

  ptr <- cpp_mlx_isclose(a$ptr, b$ptr, rtol, atol, equal_nan, device)
  .mlx_wrap_result(ptr, device)
}

#' Test if all elements of two arrays are close
#'
#' Returns a boolean scalar indicating whether all elements of two arrays
#' are close within specified tolerances.
#'
#' @param a,b MLX arrays or objects coercible to MLX arrays
#' @param rtol Relative tolerance (default: 1e-5)
#' @param atol Absolute tolerance (default: 1e-8)
#' @param equal_nan If `TRUE`, NaN values are considered equal (default: `FALSE`)
#' @inheritParams common_params
#'
#' @details
#' Two values are considered close if:
#' \code{abs(a - b) <= (atol + rtol * abs(b))}
#'
#' This function returns `TRUE` only if all elements are close.
#' Supports NumPy-style broadcasting.
#'
#' @return An mlx array containing a single boolean value
#'
#' @seealso [mlx_isclose()], [all.equal.mlx()],
#'   [mlx.core.allclose](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.allclose.html)
#' @export
#' @examples
#' a <- as_mlx(c(1.0, 2.0, 3.0))
#' b <- as_mlx(c(1.0 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-6))
#' as.logical(as.matrix(mlx_allclose(a, b)))  # TRUE
mlx_allclose <- function(a, b, rtol = 1e-5, atol = 1e-8, equal_nan = FALSE,
                         device = mlx_default_device()) {
  a <- as_mlx(a)
  b <- as_mlx(b)

  ptr <- cpp_mlx_allclose(a$ptr, b$ptr, rtol, atol, equal_nan, device)
  .mlx_wrap_result(ptr, device)
}

#' Complex-valued helpers for mlx arrays
#'
#' `mlx_real()`, `mlx_imag()`, and `mlx_conjugate()` expose MLX's complex helpers to
#' extract the real part, imaginary part, or complex conjugate of an `mlx`
#' array. Corresponding S3 methods for [Re()], [Im()], and [Conj()] are also
#' provided.
#'
#' @inheritParams mlx_array_required
#' @return An `mlx` array containing the requested component.
#' @seealso [mlx.core.array](https://ml-explore.github.io/mlx/build/html/python/array.html#complex-helpers)
#' @export
#' @examples
#' z <- as_mlx(1:4 + 1i * (4:1))
#' mlx_real(z)
#' Im(z)
mlx_real <- function(x) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_unary(x$ptr, "real")
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_real
#' @export
mlx_imag <- function(x) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_unary(x$ptr, "imag")
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_real
#' @export
mlx_conjugate <- function(x) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_unary(x$ptr, "conj")
  .mlx_wrap_result(ptr, x$device)
}

#' Convert between radians and degrees
#'
#' `mlx_degrees()` and `mlx_radians()` mirror
#' [`mlx.core.degrees()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.degrees)
#' and [`mlx.core.radians()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.radians),
#' converting angular values elementwise using MLX kernels.
#'
#' @inheritParams mlx_array_required
#' @return An mlx array with transformed angular units.
#' @seealso [mlx.core.degrees](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.degrees),
#'   [mlx.core.radians](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.radians)
#' @export
#' @examples
#' x <- as_mlx(pi / 2)
#' as.matrix(mlx_degrees(x))  # 90
mlx_degrees <- function(x) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_unary(x$ptr, "degrees")
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_degrees
#' @export
#' @examples
#' angles <- mlx_radians(as_mlx(c(0, 90, 180)))
#' as.matrix(angles)
mlx_radians <- function(x) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_unary(x$ptr, "radians")
  .mlx_wrap_result(ptr, x$device)
}

#' Detect signed infinities in mlx arrays
#'
#' `mlx_isposinf()` and `mlx_isneginf()` mirror
#' [`mlx.core.isposinf()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.isposinf)
#' and [`mlx.core.isneginf()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.isneginf),
#' returning boolean masks of positive or negative infinities.
#'
#' @inheritParams mlx_array_required
#' @return An mlx boolean array highlighting infinite entries.
#' @seealso [mlx.core.isposinf](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.isposinf),
#'   [mlx.core.isneginf](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.isneginf)
#' @export
#' @examples
#' vals <- as_mlx(c(-Inf, -1, 0, Inf))
#' as.matrix(mlx_isposinf(vals))
mlx_isposinf <- function(x) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_unary(x$ptr, "isposinf")
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_isposinf
#' @export
#' @examples
#' as.matrix(mlx_isneginf(vals))
mlx_isneginf <- function(x) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_unary(x$ptr, "isneginf")
  .mlx_wrap_result(ptr, x$device)
}

#' Elementwise NaN and infinity predicates
#'
#' `mlx_isnan()`, `mlx_isinf()`, and `mlx_isfinite()` wrap the corresponding
#' MLX elementwise predicates, returning boolean arrays.
#'
#' @inheritParams mlx_array_required
#' @return An mlx boolean array.
#' @seealso [mlx.core.isnan](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.isnan),
#'   [mlx.core.isinf](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.isinf),
#'   [mlx.core.isfinite](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.isfinite)
#' @name mlx_isnan
NULL

#' @rdname mlx_isnan
#' @export
mlx_isnan <- function(x) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_unary(x$ptr, "isnan")
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_isnan
#' @export
mlx_isinf <- function(x) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_unary(x$ptr, "isinf")
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_isnan
#' @export
mlx_isfinite <- function(x) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_unary(x$ptr, "isfinite")
  .mlx_wrap_result(ptr, x$device)
}

#' Replace NaN and infinite values with finite numbers
#'
#' `mlx_nan_to_num()` mirrors
#' [`mlx.core.nan_to_num()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.nan_to_num),
#' filling non-finite entries with user-provided finite substitutes.
#'
#' @inheritParams mlx_array_required
#' @param nan Replacement for NaN values (default `0`). Use `NULL` to retain MLX's default.
#' @param posinf Optional replacement for positive infinity.
#' @param neginf Optional replacement for negative infinity.
#' @return An mlx array with non-finite values replaced.
#' @seealso [mlx.core.nan_to_num](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.nan_to_num)
#' @export
#' @examples
#' x <- as_mlx(c(-Inf, -1, NaN, 3, Inf))
#' as.matrix(mlx_nan_to_num(x, nan = 0, posinf = 10, neginf = -10))
mlx_nan_to_num <- function(x, nan = 0, posinf = NULL, neginf = NULL) {
  x <- as_mlx(x)

  if (!is.null(nan)) {
    if (length(nan) != 1L || !is.numeric(nan)) {
      stop("nan must be a single numeric value or NULL.", call. = FALSE)
    }
    nan <- as.numeric(nan)
  }

  if (!is.null(posinf)) {
    if (length(posinf) != 1L || !is.numeric(posinf)) {
      stop("posinf must be a single numeric value or NULL.", call. = FALSE)
    }
    posinf <- as.numeric(posinf)
  }

  if (!is.null(neginf)) {
    if (length(neginf) != 1L || !is.numeric(neginf)) {
      stop("neginf must be a single numeric value or NULL.", call. = FALSE)
    }
    neginf <- as.numeric(neginf)
  }

  ptr <- cpp_mlx_nan_to_num(x$ptr, nan, posinf, neginf)
  .mlx_wrap_result(ptr, x$device)
}

#' @export
is.nan.mlx <- function(x) {
  mlx_isnan(x)
}

#' @export
is.infinite.mlx <- function(x) {
  mlx_isinf(x)
}

#' @export
is.finite.mlx <- function(x) {
  mlx_isfinite(x)
}

#' @export
Re.mlx <- function(z) {
  mlx_real(z)
}

#' @export
Im.mlx <- function(z) {
  mlx_imag(z)
}

#' @export
Conj.mlx <- function(z) {
  mlx_conjugate(z)
}

#' Test if two MLX arrays are (nearly) equal
#'
#' S3 method for `all.equal` following R semantics. Returns `TRUE` if arrays
#' are close, or a character vector describing differences if they are not.
#'
#' @param target,current MLX arrays to compare
#' @param tolerance Numeric tolerance for comparison (default: sqrt(.Machine$double.eps))
#' @param ... Additional arguments (currently ignored)
#'
#' @details
#' This method follows R's `all.equal()` semantics:
#' - Returns `TRUE` if arrays are close within tolerance
#' - Returns a character vector describing differences otherwise
#' - Checks dimensions/shapes before comparing values
#'
#' The tolerance is converted to MLX's rtol and atol parameters:
#' - rtol = tolerance
#' - atol = tolerance
#'
#' @return Either `TRUE` or a character vector describing differences
#'
#' @seealso [mlx_allclose()], [mlx_isclose()]
#' @export
#' @method all.equal mlx
#' @examples
#' a <- as_mlx(c(1.0, 2.0, 3.0))
#' b <- as_mlx(c(1.0 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-6))
#' all.equal(a, b)  # TRUE
#'
#' c <- as_mlx(c(1.0, 2.0, 10.0))
#' all.equal(a, c)  # Character vector describing difference
all.equal.mlx <- function(target, current, tolerance = sqrt(.Machine$double.eps), ...) {
  if (!is.mlx(target)) {
    return("'target' is not an mlx object")
  }
  if (!is.mlx(current)) {
    return("'current' is not an mlx object")
  }

  # Check dimensions first
  if (!identical(dim(target), dim(current))) {
    return(paste0("Arrays have different shapes: ",
                  paste(dim(target), collapse = "x"), " vs ",
                  paste(dim(current), collapse = "x")))
  }

  # Use mlx_allclose with tolerance mapped to both rtol and atol
  result <- mlx_allclose(target, current, rtol = tolerance, atol = tolerance,
                         equal_nan = FALSE, device = target$device)

  # Convert to logical
  are_close <- as.logical(as.matrix(result))

  if (are_close) {
    return(TRUE)
  } else {
    return("Arrays are not all close within tolerance")
  }
}

#' Error function and inverse error function
#'
#' `mlx_erf()` computes the error function elementwise.
#' `mlx_erfinv()` computes the inverse error function elementwise.
#'
#' @inheritParams mlx_array_required
#' @return An mlx array with the result.
#' @seealso [mlx.core.erf](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erf.html),
#'   [mlx.core.erfinv](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erfinv.html)
#' @export
#' @examples
#' x <- as_mlx(c(-1, 0, 1))
#' as.matrix(mlx_erf(x))
mlx_erf <- function(x) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_unary(x$ptr, "erf")
  .mlx_wrap_result(ptr, x$device)
}

#' @rdname mlx_erf
#' @export
#' @examples
#' p <- as_mlx(c(-0.5, 0, 0.5))
#' as.matrix(mlx_erfinv(p))
mlx_erfinv <- function(x) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_unary(x$ptr, "erfinv")
  .mlx_wrap_result(ptr, x$device)
}
