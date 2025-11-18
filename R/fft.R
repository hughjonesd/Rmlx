#' Fast Fourier transforms for MLX arrays
#'
#' `mlx_fft()`, `mlx_fft2()`, and `mlx_fftn()` wrap the MLX FFT kernels with
#' R-friendly defaults. Inputs are converted with [as_mlx()] and results are
#' returned as `mlx` arrays.
#'
#' When `device` is `NULL`, the transform runs on the input array's device,
#' falling back to [mlx_default_device()] only when coercing non-mlx inputs.
#'
#' @param x Array-like object coercible to `mlx`.
#' @param axis Optional integer axis (1-indexed) for the one-dimensional
#'   transform. Omit the argument to use the last dimension (no negative axes).
#' @param axes Optional integer vector of axes for the multi-dimensional
#'   transforms. Supply positive, 1-based axes; omit the argument to use the
#'   trailing axes (`mlx_fft()` defaults to the last axis, `mlx_fft2()` defaults
#'   to the last two axes, and `mlx_fftn()` defaults to all axes).
#' @param inverse Logical flag; if `TRUE`, compute the inverse transform. The
#'   inverse is un-normalised to match base R's `fft()`, i.e. results are
#'   multiplied by the product of the transformed axis lengths.
#' @inheritParams common_params
#'
#' @return An `mlx` array containing complex frequency coefficients.
#' @seealso [fft()], [mlx.fft](https://ml-explore.github.io/mlx/build/html/python/fft.html)
#' @export
#' @examples
#' x <- as_mlx(c(1, 2, 3, 4))
#' mlx_fft(x)
#' mlx_fft(x, inverse = TRUE)
mlx_fft <- function(x, axis, inverse = FALSE, device = NULL) {
  axis_missing <- missing(axis)
  if (!axis_missing) {
    if (is.null(axis) || length(axis) != 1L) {
      stop("`axis` must be a single positive integer.", call. = FALSE)
    }
    axes <- as.integer(axis)
  } else {
    axes <- NULL
  }
  .mlx_fft_dispatch(
    x,
    axes = axes,
    inverse = inverse,
    device = device,
    default_axes = if (axis_missing) "last" else "none"
  )
}

#' @rdname mlx_fft
#' @export
#' @examples
#' mat <- matrix(1:9, 3, 3)
#' mlx_fft2(as_mlx(mat))
mlx_fft2 <- function(x,
                     axes,
                     inverse = FALSE,
                     device = NULL) {
  axes_missing <- missing(axes)
  if (!axes_missing) {
    if (is.null(axes)) {
      stop("`axes` must be a vector of positive integers.", call. = FALSE)
    }
    axes <- as.integer(axes)
  } else {
    axes <- NULL
  }
  .mlx_fft_dispatch(
    x,
    axes = axes,
    inverse = inverse,
    device = device,
    default_axes = if (axes_missing) "last2" else "none"
  )
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
  axes_missing <- missing(axes)
  use_default <- axes_missing || is.null(axes)
  axes <- if (use_default) NULL else as.integer(axes)
  .mlx_fft_dispatch(
    x,
    axes = axes,
    inverse = inverse,
    device = device,
    default_axes = if (use_default) "all" else "none"
  )
}

.mlx_fft_dispatch <- function(x,
                              axes,
                              inverse,
                              device,
                              default_axes = c("none", "last", "last2", "all")) {
  default_axes <- match.arg(default_axes)
  mlx_x <- as_mlx(x)
  shape <- cpp_mlx_shape(mlx_x$ptr)
  ndim <- length(shape)

  axes_zero <- if (!is.null(axes)) {
    .mlx_normalize_axes(axes, mlx_x)
  } else {
    switch(
      default_axes,
      none = NULL,
      last = {
        if (ndim < 1L) {
          stop("`axis` must be supplied for 0-dimensional arrays.", call. = FALSE)
        }
        ndim - 1L
      },
      last2 = {
        if (ndim < 2L) {
          stop("Need at least two dimensions to infer fft axes.", call. = FALSE)
        }
        (ndim - 2L):(ndim - 1L)
      },
      all = {
        if (ndim == 0L) {
          integer(0)
        } else {
          seq_len(ndim) - 1L
        }
      }
    )
  }

  handle <- .mlx_resolve_device(device, mlx_x$device)
  ptr <- .mlx_eval_with_stream(handle, function(dev) {
    cpp_mlx_fft(mlx_x$ptr, axes_zero, isTRUE(inverse), dev)
  })
  result <- new_mlx(ptr, handle$device)

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
    scale <- prod(shape[scale_axes])
    result <- result * scale
  }

  result
}
