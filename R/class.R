#' Create MLX array from R object
#'
#' @param x Numeric vector, matrix, or array to convert
#' @param dtype Data type: "float32" or "float64" (default)
#' @param device Device: "gpu" (default) or "cpu"
#' @return An object of class \code{mlx}
#' @export
#' @examples
#' \dontrun{
#' x <- as_mlx(matrix(1:12, 3, 4))
#' }
as_mlx <- function(x, dtype = c("float32", "float64"), device = mlx_default_device()) {
  dtype <- match.arg(dtype)

  if (is.mlx(x)) return(x)

  # Convert to numeric and get dimensions
  if (is.vector(x) && !is.list(x)) {
    x_num <- as.numeric(x)
    dim_vec <- length(x)
  } else if (is.matrix(x) || is.array(x)) {
    x_num <- as.numeric(x)
    dim_vec <- dim(x)
  } else {
    stop("Cannot convert object of class ", class(x)[1], " to mlx")
  }

  if (is.null(dim_vec)) {
    stop("Cannot determine dimensions of input")
  }

  # Create MLX array via C++
  ptr <- cpp_mlx_from_numeric(x_num, as.integer(dim_vec), dtype, device)

  # Create S3 object
  structure(
    list(
      ptr = ptr,
      dim = as.integer(dim_vec),
      dtype = dtype,
      device = device
    ),
    class = "mlx"
  )
}

#' Force evaluation of lazy MLX operations
#'
#' @param x An \code{mlx} object
#' @return The input object (invisibly)
#' @export
mlx_eval <- function(x) {
  stopifnot(is.mlx(x))
  cpp_mlx_eval(x$ptr)
  invisible(x)
}

#' Convert MLX array to R matrix/array
#'
#' @param x An \code{mlx} object
#' @param ... Additional arguments (ignored)
#' @return A matrix or array (numeric or logical depending on dtype)
#' @export
#' @method as.matrix mlx
as.matrix.mlx <- function(x, ...) {
  mlx_eval(x)
  out <- cpp_mlx_to_r(x$ptr)
  dim(out) <- x$dim
  out
}

#' Convert MLX array to R array
#'
#' @param x An \code{mlx} object
#' @param ... Additional arguments (ignored)
#' @return A numeric array
#' @export
as.array.mlx <- function(x, ...) {
  as.matrix.mlx(x, ...)
}

#' Convert MLX array to R vector
#'
#' @param x An \code{mlx} object
#' @param mode Character string specifying the mode (ignored)
#' @return A numeric vector
#' @export
as.vector.mlx <- function(x, mode = "any") {
  # Only works for 1D arrays
  if (length(x$dim) != 1) {
    stop("Cannot convert multi-dimensional mlx array to vector. Use as.vector(as.matrix(x)) to flatten.")
  }

  as.vector(as.matrix(x))
}

#' Test if object is an MLX array
#'
#' @param x Object to test
#' @return Logical
#' @export
is.mlx <- function(x) {
  inherits(x, "mlx")
}

#' Internal constructor for mlx objects
#'
#' @param ptr External pointer to MLX array
#' @param dim Dimensions
#' @param dtype Data type
#' @param device Device
#' @keywords internal
new_mlx <- function(ptr, dim, dtype, device) {
  structure(
    list(
      ptr = ptr,
      dim = as.integer(dim),
      dtype = dtype,
      device = device
    ),
    class = "mlx"
  )
}
