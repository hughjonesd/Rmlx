#' Automatic differentiation for MLX functions
#'
#' @description
#' `mlx_grad()` computes gradients of an R function that operates on mlx
#' arrays. The function must keep all differentiable computations in MLX
#' (e.g., via `as_mlx()` and MLX operators) and return an mlx object.
#'
#' @param f An R function. Its arguments should be mlx objects, and its return
#'   value must be an mlx array (typically a scalar loss).
#' @param ... Arguments to pass to `f`. They will be coerced to mlx if needed.
#' @param argnums Indices (1-based) identifying which arguments to
#'   differentiate with respect to. Defaults to all arguments.
#' @param value Should the function value be returned alongside gradients?
#'   Set to `TRUE` to receive a list with components `value` and `grads`.
#'
#' @return When `value = FALSE` (default), a list of mlx arrays containing the
#'   gradients in the same order as `argnums`. When `value = TRUE`, a list with
#'   elements `value` (the function output as mlx) and `grads`.
#'
#' @details
#' Keep the differentiated closure inside MLX operations. Coercing arrays back
#' to base R objects (such as `as.matrix()`, `as.numeric()`, or `[[` extraction)
#' breaks the gradient tape and results in an error.
#'
#' @seealso [mlx.core.grad](https://ml-explore.github.io/mlx/build/html/python/transforms.html#mlx.core.grad),
#'   [mlx.core.value_and_grad](https://ml-explore.github.io/mlx/build/html/python/transforms.html#mlx.core.value_and_grad)
#' @examples
#' loss <- function(w, x, y) {
#'   preds <- x %*% w
#'   resids <- preds - y
#'   sum(resids * resids) / length(y)
#' }
#' x <- as_mlx(matrix(1:8, 4, 2))
#' y <- as_mlx(matrix(c(1, 3, 2, 4), 4, 1))
#' w <- as_mlx(matrix(0, 2, 1))
#' mlx_grad(loss, w, x, y)[[1]]
#' @export
mlx_grad <- function(f, ..., argnums = NULL, value = FALSE) {
  stopifnot(is.function(f))
  args <- list(...)
  if (!length(args)) {
    stop("mlx_grad() requires at least one argument.", call. = FALSE)
  }

  mlx_args <- lapply(args, function(arg) {
    as_mlx(arg)
  })

  if (is.null(argnums)) {
    argnums <- seq_along(mlx_args)
  } else {
    if (any(argnums < 1 | argnums > length(mlx_args))) {
      stop("argnums must be between 1 and the number of arguments.",
           call. = FALSE)
    }
  }

  cpp_mlx_value_grad(
    f,
    mlx_args,
    as.integer(argnums - 1L),
    isTRUE(value)
  )
}

#' @rdname mlx_grad
#' @export
#' @examples
#' loss <- function(w, x) sum((x %*% w) * (x %*% w))
#' x <- as_mlx(matrix(1:4, 2, 2))
#' w <- as_mlx(matrix(c(1, -1), 2, 1))
#' mlx_value_grad(loss, w, x)
mlx_value_grad <- function(f, ..., argnums = NULL) {
  mlx_grad(f, ..., argnums = argnums, value = TRUE)
}

#' Stop gradient propagation through an mlx array
#'
#' @inheritParams common_params
#'
#' @return A new mlx array with identical values but zero gradient.
#' @seealso [mlx.core.stop_gradient](https://ml-explore.github.io/mlx/build/html/python/transforms.html#mlx.core.stop_gradient)
#' @export
#' @examples
#' x <- as_mlx(matrix(1:4, 2, 2))
#' mlx_stop_gradient(x)
mlx_stop_gradient <- function(x) {
  x <- as_mlx(x)
  new_ptr <- cpp_mlx_stop_gradient(x$ptr)
  new_mlx(new_ptr, x$dim, x$dtype, x$device)
}
