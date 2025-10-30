#' Fast Fourier transforms for MLX arrays
#'
#' `mlx_fft()`, `mlx_fft2()`, and `mlx_fftn()` wrap the MLX FFT kernels with
#' R-friendly defaults. Inputs are converted with [as_mlx()] and results are
#' returned as `mlx` arrays.
#'
#' @param x Array-like object coercible to `mlx`.
#' @param axis Optional integer axis (1-indexed, negatives count from the end)
#'   for the one-dimensional transform.
#' @param axes Optional integer vector of axes for the multi-dimensional
#'   transforms. When `NULL`, MLX uses all axes.
#' @param inverse Logical flag; if `TRUE`, compute the inverse transform. The
#'   inverse is un-normalised to match base R's `fft()`, i.e. results are
#'   multiplied by the product of the transformed axis lengths.
#' @param device Target device or stream. Defaults to the input array's device
#'   (or [mlx_default_device()] for non-mlx inputs).
#'
#' @return An `mlx` array containing complex frequency coefficients.
#' @seealso [fft()], \url{https://ml-explore.github.io/mlx/build/html/python/fft.html}
#' @export
#' @examples
#' x <- as_mlx(c(1, 2, 3, 4))
#' mlx_fft(x)
#' mlx_fft(x, inverse = TRUE)
mlx_fft <- function(x, axis = -1L, inverse = FALSE, device = NULL) {
  if (length(axis) > 1L) {
    stop("`axis` must be a single integer.", call. = FALSE)
  }
  axes <- if (is.null(axis)) NULL else as.integer(axis)
  .mlx_fft_dispatch(x, axes = axes, inverse = inverse, device = device)
}

#' @rdname mlx_fft
#' @export
#' @examples
#' mat <- matrix(1:9, 3, 3)
#' mlx_fft2(as_mlx(mat))
mlx_fft2 <- function(x,
                     axes = c(-2L, -1L),
                     inverse = FALSE,
                     device = NULL) {
  axes <- as.integer(axes)
  .mlx_fft_dispatch(x, axes = axes, inverse = inverse, device = device)
}

#' @rdname mlx_fft
#' @export
#' @examples
#' arr <- as_mlx(array(1:8, dim = c(2, 2, 2)))
#' mlx_fftn(arr)
mlx_fftn <- function(x,
                     axes = NULL,
                     inverse = FALSE,
                     device = NULL) {
  axes <- if (is.null(axes)) NULL else as.integer(axes)
  .mlx_fft_dispatch(x, axes = axes, inverse = inverse, device = device)
}

.mlx_fft_dispatch <- function(x,
                              axes,
                              inverse,
                              device) {
  mlx_x <- as_mlx(x)
  ndim <- length(mlx_x$dim)

  axes_zero <- .mlx_normalize_axes(axes, mlx_x)
  ndim <- length(mlx_x$dim)

  handle <- .mlx_resolve_device(device, mlx_x$device)
  ptr <- .mlx_eval_with_stream(handle, function(dev) {
    cpp_mlx_fft(mlx_x$ptr, axes_zero, isTRUE(inverse), dev)
  })
  result <- .mlx_wrap_result(ptr, handle$device)

  if (isTRUE(inverse)) {
    scale_axes <- if (is.null(axes_zero)) {
      seq_len(ndim)
    } else {
      vapply(axes_zero, function(a) {
        if (a < 0L) {
          ndim + a + 1L
        } else {
          a + 1L
        }
      }, integer(1))
    }
    scale <- prod(mlx_x$dim[scale_axes])
    result <- result * scale
  }

  result
}
