
#' Subset MLX array
#'
#' MLX subsetting is like base R with a few differences:
#'
#' * `drop = FALSE` by default.
#' * Indices containing `NA`, zero, or out-of-bounds positions give an error.
#' * Numeric indices truncate toward zero, as in R. Per-axis numeric indices
#'   must be undimensioned vectors; numeric matrices are coordinate indices.
#' * Logical per-axis indices on matrices and arrays must not exceed the axis length.
#' * Explicit `NULL` indices select no elements; omitted indices select the full axis.
#' * Single indices on a 2D or higher array are only allowed for assignment.
#'   For example, if `x` is a matrix, `x[x < 0] <- 0` is
#'   OK but `subset <- x[x < 0]` is not. Use [mlx_flatten()] explicitly for
#'   subsetting.
#' * There is one exception: as in R, a single numeric matrix index selects
#'   individual elements. The number of columns must match the rank of `x`;
#'   each row gives coordinates for one element. The return value from
#'   subsetting is a flat mlx vector.
#' * `mlx` vectors, logical masks, and matrices behave the same as their R equivalents.
#' * Duplicate vector assignments like `x[c(1,1)] <- 2:3` are undefined behaviour.
#'   Coordinate matrix assignments reject duplicate rows.
#' * Character indices match against the relevant axis dimnames.
#'
#' @inheritParams common_params
#' @param ... Indices for each dimension. Provide one per axis; omitted indices
#'   select the full extent. Logical indices recycle to the dimension length.
#' @param drop Should dimensions be dropped? (default: FALSE)
#' @return The subsetted MLX object.
#' @seealso [mlx.core.take](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.take)
#' @name mlx_subset
#' @export
#' @examples
#' x <- mlx_matrix(1:9, 3, 3)
#' x[1, ]
`[.mlx` <- function(x, ..., drop = FALSE) {
  stopifnot(is_mlx(x))
  shape <- mlx_shape(x)
  dot_expr <- as.list(substitute(alist(...)))[-1]
  idx_list <- collect_indices(dot_expr, length(shape), parent.frame())

  n_indices <- nargs() - 1L
  if (! missing(drop)) n_indices <- n_indices - 1L

  result <- if (n_indices == 1L && is_matrix_index(idx_list[[1]])) {
              matrix_subset(x, idx_list[[1]])
            } else {
              if (n_indices == 1L && length(shape) > 1L &&
                  !identical(dot_expr[[1L]], quote(expr = ))) {
                stop("To use a single index, flatten first.", call. = FALSE)
              }
              vectors_subset(x, idx_list)
            }

  if (drop) result <- drop(result)
  result
}

#' Evaluate and align index expressions with dimension count
#'
#' @param dot_expr List of unevaluated index expressions from `...`.
#' @param ndim Number of dimensions expected for the target array.
#' @param env Environment in which to evaluate the expressions.
#' @return List of length `ndim` containing evaluated indices (with `NULL`
#'   placeholders for omitted axes).
#' @noRd
collect_indices <- function(dot_expr, ndim, env) {
  if (!length(dot_expr)) {
    return(vector("list", ndim))
  }

  evaluated <- lapply(dot_expr, function(expr) {
    if (identical(expr, quote(expr = ))) return(NULL)
    # A caller can forward an omitted argument as a symbol (e.g. x[i, j]).
    # Check missing() without swallowing errors from evaluating supplied indices.
    if (is.symbol(expr) && isTRUE(tryCatch(
      eval(as.call(list(base::missing, expr)), env), error = function(e) FALSE
    ))) return(NULL)
    value <- eval(expr, env)
    if (is.null(value)) integer(0) else value
  })

  if (length(evaluated) > ndim || (length(evaluated) > 1L && length(evaluated) != ndim)) {
    stop("Incorrect number of indices supplied.", call. = FALSE)
  }

  c(evaluated, vector("list", ndim - length(evaluated)))
}

matrix_subset <- function(x, idx_mat) {
  if (is_mlx(idx_mat)) idx_mat <- as_r(idx_mat)
  shape <- mlx_shape(x)

  check_matrix_index(idx_mat, shape, assign = FALSE)

  # Convert to mlx-style zero-based indices
  idx_mat <- as_mlx(trunc(idx_mat) - 1L, dtype = "int32")

  # Per-axis indices (0-based) as mlx arrays
  coord_list <- mlx_split(idx_mat, sections = ncol(idx_mat), axis = 2L)

  res <- gather_for_subset(x, coord_list)
  mlx_reshape(res, length(res))
}

vectors_subset <- function(x, idx_list) {
  shape <- mlx_shape(x)
  if (length(idx_list) != length(shape)) {
    stop("Wrong number of indices in subset.\n",
         "To use a single logical index, flatten first.")
  }

  selection <- slice_indices(idx_list, shape)
  if (!is.null(selection)) {
    if (all(selection$start == 0L) && all(selection$stop == shape)) return(x)
    ptr <- cpp_mlx_slice(x$ptr, selection$start, selection$stop, rep(1L, length(shape)))
    return(new_mlx(ptr, dimnames = subset_dimnames(x, idx_list)))
  }

  active_axes <- which(!vapply(idx_list, is.null, logical(1)))
  if (length(active_axes) == 1L) {
    axis <- active_axes[[1L]]
    idx <- normalize_index(
      idx_list[[axis]], shape[[axis]], assign = FALSE,
      names = dimnames(x)[[axis]], strict_logical = length(shape) > 1L
    )
    idx0 <- if (is_mlx(idx)) mlx_cast(idx - 1L, dtype = "int32") else idx - 1L
    idx_arg <- if (is_mlx(idx0)) idx0$ptr else as.integer(idx0)
    ptr <- cpp_mlx_take(x$ptr, idx_arg, axis - 1L)
    idx_norm <- idx_list
    idx_norm[[axis]] <- idx
    return(new_mlx(ptr, dimnames = subset_dimnames(x, idx_norm)))
  }

  idx_list <- mapply(
    function(idx, len, axis) {
      normalize_index(idx, len, assign = FALSE, names = dimnames(x)[[axis]],
                      strict_logical = length(shape) > 1L)
    },
    idx_list, shape, seq_along(shape), SIMPLIFY = FALSE
  )
  # Cast the vectors before broadcasting so no full-size index grids are materialized.
  idx_norm <- lapply(idx_list, function(idx) mlx_cast(as_mlx(idx - 1L), dtype = "int32"))
  idx_grids <- mlx_meshgrid(idx_norm, sparse = TRUE, indexing = "ij")

  out <- gather_for_subset(x, idx_grids)
  dimnames(out) <- subset_dimnames(x, idx_list)
  out
}

check_matrix_index <- function(idx_mat, shape, assign) {
  if (! is.matrix(idx_mat)) {
    stop("Non-matrix array index. Use a numeric matrix.")
  }
  if (ncol(idx_mat) != length(shape)) {
    stop("Matrix index has wrong number of columns.")
  }
  if (! is.numeric(idx_mat)) {
    stop("Non-numeric matrix index.")
  }
  idx_mat <- trunc(idx_mat)
  if (anyNA(idx_mat)) {
    stop("Matrix index contains NA values.")
  }
  if (any(idx_mat <= 0L)) {
    stop("Matrix indices must be positive.")
  }
  for (axis in seq_along(shape)) {
    if (any(idx_mat[, axis] > shape[[axis]])) {
      stop("Matrix subset index out of bounds.")
    }
  }
  if (assign && anyDuplicated(idx_mat, MARGIN = 1) > 0L) {
    stop("Matrix index contains duplicate rows in assignment")
  }
}

gather_for_subset <- function(x, idx_list) {
  ndim <- length(mlx_shape(x))
  axes <- seq_len(ndim) - 1L
  ptr <- cpp_mlx_gather(x$ptr, idx_list, axes)
  res <- new_mlx(ptr)

  # return to the same rank as x
  res_ndim <- length(mlx_shape(res))
  added_axes <- seq(from = ndim + 1L, to = res_ndim)
  res <- mlx_squeeze(res, axes = added_axes)

  res
}

# Recognize bounded, contiguous R indices without creating or evaluating MLX arrays.
# Other index types keep the general normalization path. Restrict positions to
# exactly representable float32 integers, matching as_mlx()'s default conversion.
slice_indices <- function(idx_list, shape) {
  starts <- integer(length(shape))
  stops <- as.integer(shape)
  for (axis in seq_along(shape)) {
    idx <- idx_list[[axis]]
    if (is.null(idx)) next
    if (is_mlx(idx) || !is.numeric(idx) || is.object(idx) || !is.null(dim(idx)) ||
        !length(idx) || anyNA(idx) || any(!is.finite(idx)) ||
        any(idx < 1 | idx > min(shape[[axis]], 2^24)) ||
        any(idx != trunc(idx)) || any(diff(idx) != 1)) {
      return(NULL)
    }
    starts[[axis]] <- as.integer(idx[[1L]]) - 1L
    stops[[axis]] <- as.integer(idx[[length(idx)]])
  }
  list(start = starts, stop = stops)
}
