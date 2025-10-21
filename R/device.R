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
