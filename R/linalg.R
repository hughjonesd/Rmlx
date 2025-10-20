#' Solve a system of linear equations
#'
#' @param a An \code{mlx} matrix (the coefficient matrix)
#' @param b An \code{mlx} vector or matrix (the right-hand side). If omitted,
#'   computes the matrix inverse.
#' @param ... Additional arguments (for compatibility with base::solve)
#' @return An \code{mlx} object containing the solution
#' @export
#' @method solve mlx
solve.mlx <- function(a, b = NULL, ...) {
  if (is.null(b)) {
    # No b: compute matrix inverse
    ptr <- cpp_mlx_solve(a$ptr, NULL)
    new_mlx(ptr, a$dim, a$dtype, a$device)
  } else {
    # Convert b to mlx if needed
    if (!is.mlx(b)) {
      b <- as_mlx(b, dtype = a$dtype, device = a$device)
    }

    # Solve Ax = b
    ptr <- cpp_mlx_solve(a$ptr, b$ptr)

    # Result dimensions: if b is a vector, result is a vector
    # if b is a matrix with k columns, result has same dimensions as b
    new_mlx(ptr, b$dim, a$dtype, a$device)
  }
}
