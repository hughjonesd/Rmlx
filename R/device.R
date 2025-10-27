#' Get or set default MLX device
#'
#' @param value New default device ("gpu" or "cpu"). If missing, returns current default.
#' @return Current default device (character)
#' @seealso [mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)
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
#' @inheritParams common_params
#' @seealso [mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_synchronize("gpu")
#' stream <- mlx_new_stream()
#' mlx_synchronize(stream)
mlx_synchronize <- function(device = mlx_default_device()) {
  if (.mlx_is_stream(device)) {
    cpp_mlx_synchronize_stream(device$ptr)
    return(invisible(NULL))
  }

  device <- match.arg(device, c("gpu", "cpu"))
  cpp_mlx_synchronize(device)
  invisible(NULL)
}

#' Temporarily set the default MLX device
#'
#' @inheritParams common_params
#' @param code Expression to evaluate while `device` is active.
#' @return The result of evaluating `code`.
#' @seealso [mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)
#' @export
#' @examples
#' old <- mlx_default_device()
#' with_default_device("cpu", mlx_default_device())
#' mlx_default_device(old)
with_default_device <- function(device, code) {
  device <- match.arg(device, c("gpu", "cpu"))
  old_device <- mlx_default_device()
  on.exit(mlx_default_device(old_device), add = TRUE)
  mlx_default_device(device)
  eval.parent(substitute(code))
}

#' Check if GPU backend is available
#'
#' Determines whether the GPU backend was compiled and is available.
#'
#' @return Logical: `TRUE` if GPU is available, `FALSE` if only CPU.
#' @export
#' @examples
#' if (mlx_has_gpu()) {
#'   mlx_synchronize("gpu")
#' } else {
#'   mlx_synchronize("cpu")
#' }
mlx_has_gpu <- function() {
  # Try to synchronize on GPU - if it fails, GPU is not available
  tryCatch({
    mlx_synchronize("gpu")
    TRUE
  }, error = function(e) {
    # Check if the error is about missing GPU backend
    if (grepl("gpu", e$message, ignore.case = TRUE)) {
      FALSE
    } else {
      # Some other error, re-throw it
      stop(e)
    }
  })
}

#' Get best available device
#'
#' Returns `"gpu"` if available, otherwise `"cpu"`.
#'
#' @return Character: `"gpu"` or `"cpu"`.
#' @export
#' @examples
#' device <- mlx_get_device()
#' x <- as_mlx(1:10, device = device)
mlx_get_device <- function() {
  if (mlx_has_gpu()) "gpu" else "cpu"
}
