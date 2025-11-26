
#' Subset MLX array
#'
#' MLX subsetting is like base R with a few differences:
#'
#' * `drop = FALSE` by default.
#' * Indices containing `NA` give an error.
#' * Single indices on a 2D or higher array are only allowed for assignment.
#'   For example, if `x` is a matrix, `x[x < 0] <- 0` is
#'   OK but `subset <- x[x < 0]` is not. Use [mlx_flatten()] explicitly for
#'   subsetting.
#' * There is one exception: as in R, a single numeric matrix index selects
#'   individual elements. The number of columns must match the rank of `x`;
#'   each row gives coordinates for one element. The return value from
#'   subsetting is a flat mlx vector.
#' * `mlx` vectors, logical masks, and matrices behave the same as their R equivalents.
#' * Duplicate assignments like `x[c(1,1)] <- 2:3` are undefined behaviour.
#' * Character indices are not supported as MLX has no concept
#'   of dimension names.
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
  idx_list <- .mlx_collect_indices(dot_expr, length(shape), parent.frame())

  n_indices <- nargs() - 1L
  if (! missing(drop)) n_indices <- n_indices - 1L

  result <- if (n_indices == 1L && .is_matrix_index(idx_list[[1]])) {
              .matrix_subset(x, idx_list[[1]])
            } else {
              # here, unlike with assign, we reject 1D vectors
              .vectors_subset(x, idx_list)
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
.mlx_collect_indices <- function(dot_expr, ndim, env) {
  if (!length(dot_expr)) {
    return(vector("list", ndim))
  }

  evaluated <- lapply(dot_expr, function(expr) {
    tryCatch(
      eval(expr, env),
      error = function(e) {
        msg <- conditionMessage(e)
        if (grepl("missing", msg, fixed = FALSE)) {
          return(NULL)
        }
        stop(e)
      }
    )
  })

  if (length(evaluated) > ndim) {
    stop("Incorrect number of indices supplied.", call. = FALSE)
  }

  idx_list <- vector("list", ndim)
  if (length(evaluated)) {
    for (k in seq_along(evaluated)) {
      val <- evaluated[[k]]
      if (!is.null(val)) {
        idx_list[[k]] <- val
      }
    }
  }

  idx_list
}

.matrix_subset <- function(x, idx_mat) {
  idx_mat <- as.array(idx_mat)
  shape <- mlx_shape(x)
  ndims <- length(shape)

  .check_matrix_index(idx_mat, shape, assign = FALSE)

  # Convert to mlx-style zero-based indices
  idx_mat <- as_mlx(idx_mat - 1L, dtype = "int32", device = mlx_device(x))

  # Per-axis indices (0-based) as mlx arrays
  coord_list <- mlx_split(idx_mat, sections = ncol(idx_mat), axis = 2L)

  # if (.duplicated_rows_lex(idx_mat)) {
  #   stop("Duplicate indices are not allowed in assignment.", call. = FALSE)
  # }
  res <- .gather_for_subset(x, coord_list)
  mlx_reshape(res, length(res))
}

.vectors_subset <- function(x, idx_list) {
  shape <- mlx_shape(x)
  if (length(idx_list) != length(shape)) {
    stop("Wrong number of indices in subset.\n",
         "To use a single logical index, flatten first.")
  }
  idx_list <- mapply(.normalize_index, idx_list, shape, SIMPLIFY = FALSE,
                     MoreArgs = list(assign = FALSE))
  idx_norm <- lapply(idx_list, function (x) x - 1L)

  idx_grids <- mlx_meshgrid(idx_norm, sparse = FALSE, indexing = "ij", device = x$device)
  idx_grids <- lapply(idx_grids, mlx_cast, dtype = "int32")

  .gather_for_subset(x, idx_grids)
}

.check_matrix_index <- function(idx_mat, shape, assign) {
  if (! is.matrix(idx_mat)) {
    stop("Non-matrix array index. Use a numeric matrix.")
  }
  if (ncol(idx_mat) != length(shape)) {
    stop("Matrix index has wrong number of columns.")
  }
  if (! is.numeric(idx_mat)) {
    stop("Non-numeric matrix index.")
  }
  if (any(is.na(idx_mat))) {
    stop("Matrix index contains NA values.")
  }
  if (any(idx_mat <= 0L)) {
    stop("Matrix indices must be positive.")
  }
  if (assign && anyDuplicated(idx_mat, MARGIN = 1) > 0L) {
    stop("Matrix index contains duplicate rows in assignment")
  }
}

.gather_for_subset <- function(x, idx_list) {
  ndim <- length(mlx_shape(x))
  axes <- seq_len(ndim) - 1L
  ptr <- cpp_mlx_gather(x$ptr, idx_list, axes, mlx_device(x))
  res <- new_mlx(ptr, mlx_device(x))

  # return to the same rank as x
  res_ndim <- length(mlx_shape(res))
  added_axes <- seq(from = ndim + 1L, to = res_ndim)
  res <- mlx_squeeze(res, axes = added_axes)

  res
}
