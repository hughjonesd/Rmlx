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

  # Map R function names to MLX operations
  # Most are direct matches, but some need mapping
  op_map <- c(
    "ceiling" = "ceil"
  )

  if (op %in% names(op_map)) {
    op <- op_map[op]
  }

  # Call the C++ unary operation
  ptr <- cpp_mlx_unary(x$ptr, op)
  new_mlx(ptr, x$dim, x$dtype, x$device)
}
