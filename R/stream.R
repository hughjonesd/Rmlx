#' MLX streams for asynchronous execution
#'
#' Streams provide independent execution queues on a device, allowing overlap of
#' computation and finer control over scheduling.
#'
#' @inheritParams common_params
#' @return An object of class `mlx_stream`.
#' @seealso <https://ml-explore.github.io/mlx/build/html/usage/using_streams.html>
#' @export
#' @examples
#' stream <- mlx_new_stream()
#' stream
mlx_new_stream <- function(device = mlx_default_device()) {
  device_chr <- match.arg(device, c("gpu", "cpu"))
  ptr <- cpp_mlx_stream_new(device_chr)
  .mlx_make_stream(ptr)
}

#' @rdname mlx_new_stream
#' @description `mlx_default_stream()` returns the current default stream for a device.
#' @export
mlx_default_stream <- function(device = mlx_default_device()) {
  device_chr <- match.arg(device, c("gpu", "cpu"))
  ptr <- cpp_mlx_stream_default(device_chr)
  .mlx_make_stream(ptr)
}

#' Set the default MLX stream
#'
#' @param stream An object created by [mlx_new_stream()] or [mlx_default_stream()].
#' @return Invisibly returns `stream`.
#' @export
#' @examples
#' stream <- mlx_new_stream()
#' old <- mlx_default_stream()
#' mlx_set_default_stream(stream)
#' mlx_set_default_stream(old)  # restore
mlx_set_default_stream <- function(stream) {
  stream <- .mlx_validate_stream(stream)
  cpp_mlx_set_default_stream(stream$ptr)
  invisible(stream)
}

#' @export
print.mlx_stream <- function(x, ...) {
  cat(sprintf("mlx stream [%s] index=%d\n", x$device, x$index))
  invisible(x)
}

#' @export
format.mlx_stream <- function(x, ...) {
  sprintf("<mlx_stream device=%s index=%d>", x$device, x$index)
}

.mlx_make_stream <- function(ptr) {
  device <- cpp_mlx_stream_device(ptr)
  index <- cpp_mlx_stream_index(ptr)
  structure(
    list(ptr = ptr, device = device, index = index),
    class = "mlx_stream"
  )
}

.mlx_validate_stream <- function(stream) {
  if (.mlx_is_stream(stream)) {
    return(stream)
  }
  stop("Expected an mlx_stream", call. = FALSE)
}
