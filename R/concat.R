# Internal helper to bind arrays along a given axis ---------------------------

.mlx_bind_along_axis <- function(objs, axis) {
  if (length(objs) == 1L && is.list(objs[[1L]]) && !is_mlx(objs[[1L]])) {
    objs <- objs[[1L]]
  }
  if (!length(objs)) {
    stop("No objects to bind.", call. = FALSE)
  }

  axis <- as.integer(axis)
  if (length(axis) != 1L || is.na(axis)) {
    stop("`along`/`axis` must be a single integer.", call. = FALSE)
  }

  mlx_objs <- lapply(objs, as_mlx)
  ref <- mlx_objs[[1L]]
  ref_dim <- dim(ref)
  ndim <- length(ref_dim)
  if (!ndim) {
    stop("Cannot bind scalar mlx arrays.", call. = FALSE)
  }
  if (axis < 1L || axis > ndim) {
    stop("Axis ", axis, " is out of bounds for arrays with ", ndim, " dimensions.", call. = FALSE)
  }

  for (obj in mlx_objs) {
    if (length(dim(obj)) != ndim) {
      stop("All inputs must have the same number of dimensions.", call. = FALSE)
    }
    if (!identical(dim(obj)[-axis], ref_dim[-axis])) {
      stop("Non-bound dimensions must match across all inputs.", call. = FALSE)
    }
  }

  if (length(mlx_objs) > 1L) {
    dtypes <- lapply(mlx_objs, mlx_dtype)
    dtype <- Reduce(.promote_dtype, dtypes)
    device <- Reduce(.common_device, lapply(mlx_objs, `[[`, "device"))
  } else {
    dtype <- mlx_dtype(mlx_objs[[1L]])
    device <- mlx_objs[[1L]]$device
  }

  aligned <- lapply(mlx_objs, .mlx_cast, dtype = dtype, device = device)
  ptr <- cpp_mlx_concat(aligned, axis - 1L)
  axis_lengths <- vapply(aligned, function(x) dim(x)[axis], integer(1))
  new_dim <- dim(aligned[[1L]])
  new_dim[axis] <- sum(axis_lengths)
  new_mlx(ptr, device)
}

#' Row-bind mlx arrays
#'
#' @param ... Objects to bind. mlx arrays are kept in MLX; other inputs are
#'   coerced via `as_mlx()`.
#' @param deparse.level Compatibility argument accepted for S3 dispatch; ignored.
#' @return An mlx array stacked along the first axis.
#' @details Unlike base R's `rbind()`, this function supports arrays with more
#'   than 2 dimensions and preserves all dimensions except the first (which is
#'   summed across inputs). Base R's `rbind()` flattens higher-dimensional arrays
#'   to matrices before binding.
#' @seealso [mlx.core.concatenate](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.concatenate)
#' @export
#' @examples
#' x <- mlx_matrix(1:4, 2, 2)
#' y <- mlx_matrix(5:8, 2, 2)
#' rbind(x, y)
rbind.mlx <- function(..., deparse.level = 1) {
  .mlx_bind_along_axis(list(...), axis = 1L)
}

#' Column-bind mlx arrays
#'
#' @inheritParams rbind.mlx
#' @return An mlx array stacked along the second axis.
#' @details Unlike base R's `cbind()`, this function supports arrays with more
#'   than 2 dimensions and preserves all dimensions except the second (which is
#'   summed across inputs). Base R's `cbind()` flattens higher-dimensional arrays
#'   to matrices before binding.
#' @seealso [mlx.core.concatenate](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.concatenate)
#' @export
#' @examples
#' x <- mlx_matrix(1:4, 2, 2)
#' y <- mlx_matrix(5:8, 2, 2)
#' cbind(x, y)
cbind.mlx <- function(..., deparse.level = 1) {
  .mlx_bind_along_axis(list(...), axis = 2L)
}

#' Bind mlx arrays along an axis
#'
#' @param ... One or more mlx arrays (or a single list of arrays) to combine.
#' @param along Positive integer giving the existing axis (1-indexed) along which
#'   to bind the inputs.
#'
#' @details
#' This is an MLX-backed alternative to [abind::abind()]. All inputs must share
#' the same shape on non-bound axes. The `along` axis must already exist; to
#' create a new axis use [mlx_stack()].
#'
#' @return An mlx array formed by concatenating the inputs along `along`.
#' @seealso [mlx.core.concatenate](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.concatenate)
#' @export
#' @examples
#' x <- mlx_array(1:12, c(2, 3, 2))
#' y <- mlx_array(13:24, c(2, 3, 2))
#' z <- abind(x, y, along = 3)
#' dim(z)
abind <- function(..., along = 1L) {
  .mlx_bind_along_axis(list(...), axis = along)
}
