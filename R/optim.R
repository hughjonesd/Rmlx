#' Stochastic gradient descent optimizer
#'
#' @param params List of parameters (from `mlx_parameters()`).
#' @param lr Learning rate.
#' @return An optimizer object with a `step()` method.
#' @export
mlx_optimizer_sgd <- function(params, lr = 0.01) {
  if (!all(vapply(params, inherits, logical(1), "mlx_param"))) {
    stop("params must be a list of mlx_param objects.", call. = FALSE)
  }
  lr <- as.numeric(lr)
  if (!is.finite(lr) || lr <= 0) {
    stop("Learning rate must be positive.", call. = FALSE)
  }

  step <- function(grads) {
    if (length(grads) != length(params)) {
      stop("Gradient list length does not match parameters.", call. = FALSE)
    }
    for (i in seq_along(params)) {
      p_val <- mlx_param_get(params[[i]])
      update <- p_val - lr * grads[[i]]
      mlx_param_set(params[[i]], update)
    }
    invisible(TRUE)
  }

  structure(
    list(step = step, lr = lr, params = params),
    class = "mlx_optimizer_sgd"
  )
}

#' Single training step helper
#'
#' @param module An `mlx_module`.
#' @param loss_fn Function of `module` and data returning an `mlx` scalar.
#' @param optimizer Optimizer object from `mlx_optimizer_sgd()`.
#' @param ... Additional data passed to `loss_fn`.
#' @return A list with the current loss.
#' @export
mlx_train_step <- function(module, loss_fn, optimizer, ...) {
  if (!inherits(module, "mlx_module")) {
    stop("module must inherit from mlx_module.", call. = FALSE)
  }
  if (!is.function(loss_fn)) {
    stop("loss_fn must be a function.", call. = FALSE)
  }
  if (!inherits(optimizer, "mlx_optimizer_sgd")) {
    stop("Currently only mlx_optimizer_sgd optimizers are supported.", call. = FALSE)
  }

  params <- optimizer$params
  param_values <- mlx_param_values(params)
  data_args <- list(...)
  n_param <- length(param_values)

  loss_wrapper <- function(...) {
    dots <- list(...)
    if (length(dots) < n_param) {
      stop("Insufficient arguments passed to loss_wrapper.", call. = FALSE)
    }
    param_inputs <- dots[seq_len(n_param)]
    data_inputs <- dots[-seq_len(n_param)]
    mlx_param_set_values(params, param_inputs)
    loss <- do.call(loss_fn, c(list(module), data_inputs))
    if (!is.mlx(loss)) {
      stop("loss_fn must return an `mlx` tensor.", call. = FALSE)
    }
    loss
  }

  grad_inputs <- c(param_values, data_args)
  argnums <- seq_len(n_param)

  vg <- do.call(
    mlx_value_grad,
    c(list(loss_wrapper), grad_inputs, list(argnums = argnums))
  )

  optimizer$step(vg$grads)

  list(loss = vg$value)
}
