#' Math operations for MLX arrays
#'
#' @inheritParams mlx_array_required
#' @param ... Additional arguments (ignored)
#' @return An mlx object with the result
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/array.html}
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
#'   \url{https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.isclose.html}
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
#'   \url{https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.allclose.html}
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
