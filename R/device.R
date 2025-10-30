#' Get or set default MLX device
#'
#' @param value New default device ("gpu" or "cpu"). If missing, returns current default.
#' @return Current default device (character)
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/metal.html}
#' @export
#' @examples
#' mlx_default_device()  # Get current default
#' mlx_default_device("cpu")  # Set to CPU
#' mlx_default_device("gpu")  # Set back to GPU
#' mlx_default_device()
mlx_default_device <- function(value) {
  if (missing(value)) {
    return(cpp_mlx_default_device())
  }
  value <- match.arg(value, c("gpu", "cpu"))
  cpp_mlx_set_default_device(value)
  value
}

#' Synchronize MLX execution
#'
#' Waits for outstanding operations on the specified device or stream to complete.
#'
#' @param device Device identifier ("gpu" or "cpu") or an `mlx_stream` created by
#'   [mlx_new_stream()].
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/metal.html}
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_synchronize("gpu")
#' stream <- mlx_new_stream()
#' mlx_synchronize(stream)
mlx_synchronize <- function(device = c("gpu", "cpu")) {
  if (.mlx_is_stream(device)) {
    cpp_mlx_synchronize_stream(device$ptr)
    return(invisible(NULL))
  }

  device <- match.arg(device)
  cpp_mlx_synchronize(device)
  invisible(NULL)
}

#' Temporarily set the default MLX device
#'
#' @param device Device to use (`"gpu"` or `"cpu"`).
#' @param code Expression to evaluate while `device` is active.
#' @return The result of evaluating `code`.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/metal.html}
#' @export
#' @examples
#' old <- mlx_default_device()
#' with_default_device("cpu", mlx_default_device())
#' mlx_default_device(old)
with_default_device <- function(device = c("gpu", "cpu"), code) {
  device <- match.arg(device)
  old_device <- mlx_default_device()
  on.exit(mlx_default_device(old_device), add = TRUE)
  mlx_default_device(device)
  eval.parent(substitute(code))
}
