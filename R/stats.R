#' Row means generic
#'
#' @param x An array
#' @param ... Additional arguments
#' @export
rowMeans <- function(x, ...) {
  UseMethod("rowMeans")
}

#' Default rowMeans method
#' @export
rowMeans.default <- base::rowMeans

#' Column means generic
#'
#' @param x An array
#' @param ... Additional arguments
#' @export
colMeans <- function(x, ...) {
  UseMethod("colMeans")
}

#' Default colMeans method
#' @export
colMeans.default <- base::colMeans

#' Sum of MLX array elements
#'
#' @param x An \code{mlx} object
#' @param ... Additional arguments (ignored)
#' @param na.rm Ignored (for compatibility)
#' @return An \code{mlx} scalar
#' @export
#' @method sum mlx
sum.mlx <- function(x, ..., na.rm = FALSE) {
  .mlx_reduce(x, "sum")
}

#' Mean of MLX array elements
#'
#' @param x An \code{mlx} object
#' @param ... Additional arguments (ignored)
#' @return An \code{mlx} scalar
#' @export
#' @method mean mlx
mean.mlx <- function(x, ...) {
  .mlx_reduce(x, "mean")
}

#' Column means of MLX matrix
#'
#' @param x An \code{mlx} matrix
#' @param na.rm Ignored (for compatibility)
#' @param dims Which dimensions are regarded as columns (default: 1)
#' @param ... Additional arguments (ignored)
#' @return An \code{mlx} vector with column means
#' @export
#' @method colMeans mlx
colMeans.mlx <- function(x, na.rm = FALSE, dims = 1, ...) {
  # In R's column-major layout, columns are along axis 1 (after transposing MLX view)
  # MLX uses row-major, so we need to think carefully about axis mapping
  # For a standard R matrix, colMeans operates on columns (axis=0 in row-major)
  .mlx_reduce_axis(x, "mean", axis = 0L, keepdims = FALSE)
}

#' Row means of MLX matrix
#'
#' @param x An \code{mlx} matrix
#' @param na.rm Ignored (for compatibility)
#' @param dims Which dimensions are regarded as rows (default: 1)
#' @param ... Additional arguments (ignored)
#' @return An \code{mlx} vector with row means
#' @export
#' @method rowMeans mlx
rowMeans.mlx <- function(x, na.rm = FALSE, dims = 1, ...) {
  # Row means operate on rows (axis=1 in row-major)
  .mlx_reduce_axis(x, "mean", axis = 1L, keepdims = FALSE)
}

#' Row sums generic
#'
#' @param x An array
#' @param ... Additional arguments
#' @export
rowSums <- function(x, ...) {
  UseMethod("rowSums")
}

#' Default rowSums method
#' @export
rowSums.default <- base::rowSums

#' Row sums of MLX matrix
#'
#' @param x An \code{mlx} matrix
#' @param na.rm Ignored (for compatibility)
#' @param dims Which dimensions are regarded as rows (default: 1)
#' @param ... Additional arguments (ignored)
#' @return An \code{mlx} vector with row sums
#' @export
#' @method rowSums mlx
rowSums.mlx <- function(x, na.rm = FALSE, dims = 1, ...) {
  # Row sums operate on rows (axis=1 in row-major)
  .mlx_reduce_axis(x, "sum", axis = 1L, keepdims = FALSE)
}

#' Column sums generic
#'
#' @param x An array
#' @param ... Additional arguments
#' @export
colSums <- function(x, ...) {
  UseMethod("colSums")
}

#' Default colSums method
#' @export
colSums.default <- base::colSums

#' Column sums of MLX matrix
#'
#' @param x An \code{mlx} matrix
#' @param na.rm Ignored (for compatibility)
#' @param dims Which dimensions are regarded as columns (default: 1)
#' @param ... Additional arguments (ignored)
#' @return An \code{mlx} vector with column sums
#' @export
#' @method colSums mlx
colSums.mlx <- function(x, na.rm = FALSE, dims = 1, ...) {
  # Column sums operate on columns (axis=0 in row-major)
  .mlx_reduce_axis(x, "sum", axis = 0L, keepdims = FALSE)
}

#' Transpose of MLX matrix
#'
#' @param x An \code{mlx} matrix
#' @return Transposed \code{mlx} matrix
#' @export
#' @method t mlx
t.mlx <- function(x) {
  # Must transpose in MLX so MLX shape matches R dims
  # Layout conversion (physical reordering) happens at boundaries during copy
  ptr <- cpp_mlx_transpose(x$ptr)
  new_mlx(ptr, rev(x$dim), x$dtype, x$device)
}

#' Cross product
#'
#' @param x An \code{mlx} matrix
#' @param y An \code{mlx} matrix (default: NULL, uses x)
#' @return \code{t(x) \%*\% y} as an \code{mlx} object
#' @export
#' @method crossprod mlx
crossprod.mlx <- function(x, y = NULL) {
  if (is.null(y)) y <- x
  t(x) %*% y
}

#' Transposed cross product
#'
#' @param x An \code{mlx} matrix
#' @param y An \code{mlx} matrix (default: NULL, uses x)
#' @return \code{x \%*\% t(y)} as an \code{mlx} object
#' @export
#' @method tcrossprod mlx
tcrossprod.mlx <- function(x, y = NULL) {
  if (is.null(y)) y <- x
  x %*% t(y)
}

# Internal helper: full reduction
.mlx_reduce <- function(x, op) {
  ptr <- cpp_mlx_reduce(x$ptr, op)
  # Result is a scalar (0-d array or 1-element array)
  new_mlx(ptr, 1L, x$dtype, x$device)
}

# Internal helper: axis reduction
.mlx_reduce_axis <- function(x, op, axis, keepdims) {
  ptr <- cpp_mlx_reduce_axis(x$ptr, op, as.integer(axis), keepdims)

  # Calculate new dimensions
  if (keepdims) {
    new_dim <- x$dim
    new_dim[axis + 1L] <- 1L  # R is 1-indexed
  } else {
    new_dim <- x$dim[-(axis + 1L)]  # Remove the reduced dimension
    if (length(new_dim) == 0) new_dim <- 1L
  }

  new_mlx(ptr, new_dim, x$dtype, x$device)
}
