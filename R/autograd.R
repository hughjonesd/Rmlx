#' Automatic differentiation for MLX functions
#'
#' @description
#' `mlx_grad()` computes gradients of an R function that operates on `mlx`
#' tensors. The function must keep all differentiable computations in MLX
#' (e.g., via `as_mlx()` and MLX operators) and return an `mlx` object.
#'
#' @param f An R function. Its arguments should be `mlx` objects, and its return
#'   value must be an `mlx` tensor (typically a scalar loss).
#' @param ... Arguments to pass to `f`. They will be coerced to `mlx` if needed.
#' @param argnums Indices (1-based) identifying which arguments to
#'   differentiate with respect to. Defaults to all arguments.
#' @param value Should the function value be returned alongside gradients?
#'   Set to `TRUE` to receive a list with components `value` and `grads`.
#'
#' @return When `value = FALSE` (default), a list of `mlx` tensors containing the
#'   gradients in the same order as `argnums`. When `value = TRUE`, a list with
#'   elements `value` (the function output as `mlx`) and `grads`.
#'
#' @examples
#' \dontrun{
#' loss <- function(w, x, y) {
#'   preds <- x %*% w
#'   resids <- preds - y
#'   sum(resids * resids) / length(y)
#' }
#' x <- as_mlx(matrix(rnorm(20), 5, 4))
#' y <- as_mlx(matrix(rnorm(5), 5, 1))
#' w <- as_mlx(matrix(0, 4, 1))
#' grad_w <- mlx_grad(loss, w, x, y)[[1]]
#' }
#' @export
mlx_grad <- function(f, ..., argnums = NULL, value = FALSE) {
  stopifnot(is.function(f))
  args <- list(...)
  if (!length(args)) {
    stop("mlx_grad() requires at least one argument.", call. = FALSE)
  }

  mlx_args <- lapply(args, function(arg) {
    if (is.mlx(arg)) arg else as_mlx(arg)
  })

  if (is.null(argnums)) {
    argnums <- seq_along(mlx_args)
  } else {
    if (any(argnums < 1 | argnums > length(mlx_args))) {
      stop("argnums must be between 1 and the number of arguments.",
           call. = FALSE)
    }
  }

  res <- cpp_mlx_value_grad(
    f,
    mlx_args,
    as.integer(argnums - 1L),
    isTRUE(value)
  )

  if (isTRUE(value)) {
    res$value <- structure(res$value, class = class(res$value))
    res$grads <- res$grads
    return(res)
  }

  res
}

#' @rdname mlx_grad
#' @export
mlx_value_grad <- function(f, ..., argnums = NULL) {
  mlx_grad(f, ..., argnums = argnums, value = TRUE)
}

#' Stop gradient propagation through an MLX tensor
#'
#' @param x An `mlx` tensor.
#'
#' @return A new `mlx` tensor with identical values but zero gradient.
#' @export
mlx_stop_gradient <- function(x) {
  if (!is.mlx(x)) x <- as_mlx(x)
  new_ptr <- cpp_mlx_stop_gradient(x$ptr)
  new_mlx(new_ptr, x$dim, x$dtype, x$device)
}
