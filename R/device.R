#' Get or set default MLX device
#'
#' @param value New default device ("gpu" or "cpu"). If missing, returns current default.
#' @return Current default device (character)
#' @export
#' @examples
#' mlx_default_device()  # Get current default
#' mlx_default_device("cpu")  # Set to CPU
#' mlx_default_device("gpu")  # Set back to GPU
#' mlx_default_device()
mlx_default_device <- local({
  dev <- "gpu"
  function(value) {
    if (!missing(value)) {
      dev <<- match.arg(value, c("gpu", "cpu"))
    }
    dev
  }
})

#' Synchronize MLX device
#'
#' Waits for all outstanding operations on the specified device to complete.
#'
#' @param device Device to synchronize ("gpu" or "cpu").
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_synchronize("gpu")
mlx_synchronize <- function(device = c("gpu", "cpu")) {
  device <- match.arg(device)
  cpp_mlx_synchronize(device)
  invisible(NULL)
}

#' Temporarily set the default MLX device
#'
#' @param device Device to use (`"gpu"` or `"cpu"`).
#' @param code Expression to evaluate while `device` is active.
#' @return The result of evaluating `code`.
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
