#' Get or set default MLX device
#'
#' @param value New default device ("gpu" or "cpu"). If missing, returns the
#'   current default.
#' @return Current default device (character).
#' @seealso [mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)
#' @export
#' @examples
#' mlx_default_device()  # Get current default
#' mlx_default_device("cpu")  # Set to CPU
#' if (mlx_has_gpu()) {
#'   mlx_default_device("gpu")  # Set back to GPU
#'   mlx_default_device()
#' }
#' mlx_default_device("cpu")
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
#' @return Returns `NULL` invisibly.
#' @seealso [mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)
#' @export
#' @examples
#' x <- mlx_matrix(1:4, 2, 2)
#' mlx_synchronize("cpu")
#' if (mlx_has_gpu()) mlx_synchronize("gpu")
#' stream <- mlx_new_stream()
#' mlx_synchronize(stream)
mlx_synchronize <- function(device = mlx_default_device()) {
  if (is_mlx_stream(device)) {
    cpp_mlx_synchronize_stream(device$ptr)
    return(invisible(NULL))
  }

  device <- match.arg(device, c("gpu", "cpu"))
  cpp_mlx_synchronize(device)
  invisible(NULL)
}

#' Temporarily set the default MLX device or stream
#'
#' @param device `"gpu"`, `"cpu"`, or an `mlx_stream` created via [mlx_new_stream()].
#' @param code Expression to evaluate while `device` is active.
#' @return The result of evaluating `code`.
#' @seealso [mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)
#' @export
#' @examples
#' with_default_device("cpu", x <- mlx_vector(1:10))
#'
with_default_device <- function(device, code) {
  if (is_mlx_stream(device)) {
    stream <- .mlx_validate_stream(device)
    target_device <- stream$device
    old_device <- mlx_default_device()
    old_stream <- mlx_default_stream(target_device)
    on.exit({
      mlx_set_default_stream(old_stream)
      mlx_default_device(old_device)
    }, add = TRUE)
    mlx_default_device(target_device)
    mlx_set_default_stream(stream)
    return(eval.parent(substitute(code)))
  }

  device_chr <- match.arg(device, c("gpu", "cpu"))
  old_device <- mlx_default_device()
  on.exit(mlx_default_device(old_device), add = TRUE)
  mlx_default_device(device_chr)
  eval.parent(substitute(code))
}

#' Set the default MLX device for the current scope
#'
#' Use `local_default_device()` to temporarily switch devices within the current
#' function.
#'
#' @param .local_envir Environment to bind the restoration to. Defaults to the
#'   calling environment.
#' @return Invisibly returns the previous default device.
#' @export
#' @rdname with_default_device
#' @examples
#' local_default_device("cpu")
#' # code here runs on CPU, then the previous default is restored
local_default_device <- function(device, .local_envir = parent.frame()) {
  if (is_mlx_stream(device)) {
    stream <- .mlx_validate_stream(device)
    target_device <- stream$device
    old_device <- mlx_default_device()
    old_stream <- mlx_default_stream(target_device)
    do.call(
      on.exit,
      list(substitute({
        mlx_set_default_stream(old_stream_val)
        mlx_default_device(old_device_val)
      }, list(
        old_stream_val = old_stream,
        old_device_val = old_device
      )), add = TRUE),
      envir = .local_envir
    )
    mlx_default_device(target_device)
    mlx_set_default_stream(stream)
    return(invisible(old_device))
  }

  device_chr <- match.arg(device, c("gpu", "cpu"))
  old_device <- mlx_default_device()
  do.call(
    on.exit,
    list(substitute(mlx_default_device(old_device_val),
      list(old_device_val = old_device)
    ), add = TRUE),
    envir = .local_envir
  )
  mlx_default_device(device_chr)
  invisible(old_device)
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
  cpp_mlx_has_gpu()
}

#' Get best available device
#'
#' Returns `"gpu"` if available, otherwise `"cpu"`.
#'
#' @return Character: `"gpu"` or `"cpu"`.
#' @export
#' @examples
#' device <- mlx_best_device()
#' x <- as_mlx(1:10, device = device)
mlx_best_device <- function() {
  if (mlx_has_gpu()) "gpu" else "cpu"
}

#' Get device associated with an MLX object
#'
#' @inheritParams common_params
#' @return `"gpu"` or `"cpu"`.
#' @export
#' @examples
#' x <- as_mlx(1:10)
#' mlx_device(x)
mlx_device <- function(x) {
  stopifnot(is_mlx(x))
  x$device
}
