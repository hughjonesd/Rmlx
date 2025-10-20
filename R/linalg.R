#' Solve a system of linear equations
#'
#' @param a An `mlx` matrix (the coefficient matrix)
#' @param b An `mlx` vector or matrix (the right-hand side). If omitted,
#'   computes the matrix inverse.
#' @param ... Additional arguments (for compatibility with base::solve)
#' @return An `mlx` object containing the solution
#' @export
#' @method solve mlx
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
