#' Math operations for MLX arrays
#'
#' @param x An \code{mlx} object
#' @param ... Additional arguments (ignored)
#' @return An \code{mlx} object with the result
#' @export
#' @method Math mlx
Math.mlx <- function(x, ...) {
  # .Generic contains the name of the function that was called
  op <- .Generic

  # Cumulative operations flatten the array in column-major order (like R)
  # MLX flattens in row-major order, so we need to fall back to R
  if (op %in% c("cumsum", "cumprod", "cummax", "cummin")) {
    x_r <- as.matrix(x)
    result_r <- get(.Generic, mode = "function")(x_r, ...)
    return(as_mlx(result_r, dtype = x$dtype, device = x$device))
  }

  # Map R function names to MLX operations
  # Most are direct matches, but some need mapping
  op_map <- c(
    "ceiling" = "ceil"
  )

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
