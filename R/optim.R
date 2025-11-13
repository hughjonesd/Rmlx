#' Stochastic gradient descent optimizer
#'
#' @param params List of parameters (from `mlx_parameters()`).
#' @param lr Learning rate.
#' @return An optimizer object with a `step()` method.
#' @seealso [mlx.optimizers.SGD](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.optimizers.SGD)
#' @export
#' @examples
#' set.seed(1)
#' model <- mlx_linear(2, 1, bias = FALSE)
#' opt <- mlx_optimizer_sgd(mlx_parameters(model), lr = 0.1)
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
    param_vals <- mlx_param_values(params)
    updates <- Map(function(p_val, g) p_val - lr * g, param_vals, grads)
    mlx_param_set_values(params, updates)
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
#' @param loss_fn Function of `module` and data returning an mlx scalar.
#' @param optimizer Optimizer object from `mlx_optimizer_sgd()`.
#' @param ... Additional data passed to `loss_fn`.
#' @return A list with the current loss.
#' @seealso [mlx.optimizers.Optimizer](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.optimizers.Optimizer)
#' @export
#' @examples
#' set.seed(1)
#' model <- mlx_linear(2, 1, bias = FALSE)
#' opt <- mlx_optimizer_sgd(mlx_parameters(model), lr = 0.1)
#' data_x <- as_mlx(matrix(c(1, 2, 3, 4), 2, 2))
#' data_y <- as_mlx(matrix(c(5, 6), 2, 1))
#' loss_fn <- function(model, x, y) {
#'   pred <- model$forward(x)
#'   mean((pred - y)^2)
#' }
#' result <- mlx_train_step(model, loss_fn, opt, data_x, data_y)
mlx_train_step <- function(module, loss_fn, optimizer, ...) {
  params <- optimizer$params
  n_param <- length(params)
  param_values <- mlx_param_values(params)

  dots <- list(...)
  data_args <- dots

  loss_wrapper <- function(...) {
    args <- list(...)
    param_inputs <- args[seq_len(n_param)]
    data_inputs <- args[-seq_len(n_param)]
    mlx_param_set_values(params, param_inputs)
    loss <- do.call(loss_fn, c(list(module), data_inputs))
    if (!is.mlx(loss)) {
      stop("loss_fn must return an mlx array.", call. = FALSE)
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

#' Coordinate Descent with L1 Regularization
#'
#' Minimizes f(beta) + lambda * ||beta||_1 using coordinate descent,
#' where f is a smooth differentiable loss function.
#'
#' @param loss_fn Function(beta) -> scalar loss (MLX tensor). Must be smooth and differentiable.
#' @param beta_init Initial parameter vector (p x 1 MLX tensor).
#' @param lambda L1 penalty parameter (scalar, default 0).
#' @param grad_fn Optional gradient function. If NULL, computed via mlx_grad(loss_fn).
#' @param lipschitz Optional Lipschitz constants for each coordinate (length p vector).
#'   If NULL, uses simple constant estimates.
#' @param batch_size Number of coordinates to update per iteration (default: adaptive based on p).
#'   - 1 = pure coordinate descent (sequential)
#'   - p = full batch (all coordinates updated together)
#'   - intermediate values = mini-batch coordinate descent
#' @param compile Whether to compile the update step (default FALSE, not yet implemented).
#' @param max_iter Maximum number of iterations (default 1000).
#' @param tol Convergence tolerance (default 1e-6).
#'
#' @return List with:
#'   - beta: Optimized parameter vector (MLX tensor)
#'   - n_iter: Number of iterations performed
#'   - converged: Whether convergence criterion was met
#'
#' @details
#' This function implements proximal coordinate descent for problems of the form:
#'   min_beta f(beta) + lambda * ||beta||_1
#'
#' where f is smooth. At each iteration, coordinates are updated via the proximal gradient step:
#'   z_j = beta_j - (1/L_j) * grad_f(beta)_j
#'   beta_j = soft_threshold(z_j, lambda/L_j)
#'
#' where L_j is a Lipschitz constant for coordinate j.
#'
#' Batching updates multiple coordinates simultaneously, which can significantly improve
#' performance by reducing R-to-MLX call overhead.
#'
#' @export
#' @examples
#' # Lasso regression: min 0.5*||y - X*beta||^2 + lambda*||beta||_1
#' n <- 100
#' p <- 50
#' X <- as_mlx(matrix(rnorm(n*p), n, p))
#' y <- as_mlx(matrix(rnorm(n), ncol=1))
#' beta_init <- mlx_zeros(c(p, 1))
#'
#' loss_fn <- function(beta) {
#'   residual <- y - X %*% beta
#'   sum(residual^2) / (2*n)
#' }
#'
#' result <- mlx_coordinate_descent(
#'   loss_fn = loss_fn,
#'   beta_init = beta_init,
#'   lambda = 0.1,
#'   batch_size = 10
#' )
mlx_coordinate_descent <- function(loss_fn,
                                    beta_init,
                                    lambda = 0,
                                    grad_fn = NULL,
                                    lipschitz = NULL,
                                    batch_size = NULL,
                                    compile = FALSE,
                                    max_iter = 1000,
                                    tol = 1e-6) {

  if (!is.mlx(beta_init)) {
    stop("beta_init must be an MLX array", call. = FALSE)
  }

  beta <- beta_init
  n_pred <- nrow(beta)

  # Default batch size: sequential for small p, batched for large p
  if (is.null(batch_size)) {
    batch_size <- if (n_pred <= 100) 1 else min(50, n_pred)
  }

  # Gradient computation function
  compute_grad <- if (is.null(grad_fn)) {
    function(beta) mlx_grad(loss_fn, beta)[[1]]
  } else {
    grad_fn
  }

  # Default Lipschitz constants
  use_backtracking <- is.null(lipschitz)
  if (!use_backtracking) {
    lipschitz <- as.numeric(lipschitz)
    if (length(lipschitz) != n_pred) {
      stop("lipschitz must have length equal to number of parameters", call. = FALSE)
    }
  } else {
    # Conservative default
    lipschitz <- rep(1.0, n_pred)
  }

  # Create coordinate batches
  coord_batches <- split(seq_len(n_pred), ceiling(seq_len(n_pred) / batch_size))

  for (iter in seq_len(max_iter)) {
    beta_old <- as.numeric(beta)

    # Cycle through coordinate batches
    for (coords in coord_batches) {
      # Compute gradient
      grad <- compute_grad(beta)

      # Update all coordinates in the batch at once
      grad_coords <- grad[coords, , drop = FALSE]
      beta_coords <- beta[coords, , drop = FALSE]
      L_coords <- lipschitz[coords]

      # Convert L_coords to mlx column vector for proper broadcasting
      L_mlx <- as_mlx(L_coords)
      L_matrix <- mlx_reshape(L_mlx, c(length(L_coords), 1))

      # Proximal gradient step (vectorized)
      z <- beta_coords - grad_coords / L_matrix

      # Soft thresholding (vectorized)
      abs_z <- abs(z)
      threshold_mlx <- mlx_reshape(as_mlx(lambda / L_coords), c(length(L_coords), 1))

      # Apply soft thresholding: max(abs_z - threshold, 0) * sign(z)
      beta[coords, ] <- sign(z) * mlx_maximum(abs_z - threshold_mlx, 0)
    }

    # Check convergence
    delta <- max(abs(as.numeric(beta) - beta_old))
    if (delta < tol) {
      return(list(
        beta = beta,
        n_iter = iter,
        converged = TRUE
      ))
    }
  }

  list(
    beta = beta,
    n_iter = max_iter,
    converged = FALSE
  )
}
