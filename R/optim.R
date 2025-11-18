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
#' @param max_iter Maximum number of iterations (default 1000).
#' @param tol Convergence tolerance (default 1e-6).
#' @param block_size Number of coordinates to update before recomputing the gradient.
#'   Set to 1 for strict coordinate descent; larger values trade accuracy for speed.
#' @param ridge_penalty Optional ridge (L2) penalty term applied per-coordinate when
#'   computing gradients. Can be a scalar, numeric vector of length p, or an `mlx`
#'   array with shape compatible with `beta_init`.
#' @param grad_cache Optional environment for supplying cached gradient components.
#'   Supported fields are `type = "gaussian"` with entries `x`, `residual`, `n_obs`,
#'   and optional `ridge_penalty`; or `type = "binomial"` with entries `x`, `eta`,
#'   `mu`, `residual`, `y`, `n_obs`, and optional `ridge_penalty`.
#'
#' @return List with:
#'   - beta: Optimized parameter vector (MLX tensor)
#'   - n_iter: Number of iterations performed
#'   - converged: Whether convergence criterion was met
#'
#' @details
#' This function implements proximal gradient descent for problems of the form:
#'   min_beta f(beta) + lambda * ||beta||_1
#'
#' where f is smooth. At each iteration, all coordinates are updated via:
#'   z = beta - (1/L) * grad_f(beta)
#'   beta = soft_threshold(z, lambda/L)
#'
#' where L are Lipschitz constants for each coordinate.
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
#'   block_size = 8
#' )
#'
#' # Reuse cached residuals for a Gaussian problem
#' grad_cache <- new.env(parent = emptyenv())
#' grad_cache$type <- "gaussian"
#' grad_cache$x <- X
#' grad_cache$n_obs <- n
#' grad_cache$residual <- y - X %*% beta_init
#' cached <- mlx_coordinate_descent(
#'   loss_fn = loss_fn,
#'   beta_init = beta_init,
#'   lambda = 0.1,
#'   grad_cache = grad_cache
#' )
mlx_coordinate_descent <- function(loss_fn,
                                    beta_init,
                                    lambda = 0,
                                    ridge_penalty = 0,
                                    grad_fn = NULL,
                                    lipschitz = NULL,
                                    max_iter = 1000,
                                    tol = 1e-6,
                                    block_size = 1,
                                    grad_cache = NULL) {

  beta <- as_mlx(beta_init)
  n_pred <- nrow(beta)

  # Gradient computation function
  compute_grad <- if (is.null(grad_fn)) {
    function(beta) mlx_grad(loss_fn, beta)[[1]]
  } else {
    grad_fn
  }

  # Default Lipschitz constants
  if (is.null(lipschitz)) {
    lips_numeric <- rep(1.0, n_pred)
  } else {
    lips_numeric <- as.numeric(lipschitz)
    if (length(lips_numeric) != n_pred) {
      stop("lipschitz must have length equal to number of parameters", call. = FALSE)
    }
  }

  if (!all(is.finite(lips_numeric)) || any(lips_numeric <= 0)) {
    stop("All Lipschitz constants must be finite and positive.", call. = FALSE)
  }

  if (!is.numeric(lambda) || length(lambda) != 1L) {
    stop("lambda must be a numeric scalar.", call. = FALSE)
  }
  lambda_numeric <- as.numeric(lambda)
  if (!is.finite(lambda_numeric) || lambda_numeric < 0) {
    stop("lambda must be finite and non-negative.", call. = FALSE)
  }

  block_size <- as.integer(block_size)
  if (is.na(block_size) || block_size < 1L) {
    block_size <- 1L
  }
  block_size <- min(block_size, n_pred)

  lipschitz_mlx <- if (is.mlx(lipschitz)) {
    arr <- lipschitz
    if (length(cpp_mlx_shape(arr$ptr)) == 1L) {
      mlx_reshape(arr, c(n_pred, 1))
    } else {
      arr
    }
  } else {
    mlx_reshape(as_mlx(lips_numeric), c(n_pred, 1))
  }
  lambda_mlx <- as_mlx(matrix(lambda_numeric, nrow = 1, ncol = 1))

  as_column_mlx <- function(val) {
    if (is.mlx(val)) {
      arr <- val
      if (length(cpp_mlx_shape(arr$ptr)) == 1L) {
        return(mlx_reshape(arr, c(n_pred, 1)))
      }
      return(arr)
    }
    vec <- as.numeric(val)
    if (length(vec) == 0L) {
      vec <- 0
    }
    if (length(vec) == 1L) {
      vec <- rep(vec, n_pred)
    } else if (length(vec) != n_pred) {
      stop("ridge_penalty must be a scalar or length n_pred.", call. = FALSE)
    }
    mlx_reshape(as_mlx(vec), c(n_pred, 1))
  }

  ridge_mlx <- as_column_mlx(ridge_penalty)

  blocks <- split(seq_len(n_pred), ceiling(seq_len(n_pred) / block_size))

  for (iter in seq_len(max_iter)) {
    beta_prev <- beta

    for (block in blocks) {
      if (!is.null(grad_cache) && grad_cache$type %in% c("gaussian", "binomial")) {
        x_block <- grad_cache$x[, block, drop = FALSE]
        if (identical(grad_cache$type, "gaussian")) {
          grad_block <- -crossprod(x_block, grad_cache$residual) / grad_cache$n_obs
        } else {
          grad_block <- crossprod(x_block, grad_cache$residual) / grad_cache$n_obs
        }
        ridge_val <- if (!is.null(grad_cache$ridge_penalty)) {
          as_column_mlx(grad_cache$ridge_penalty)
        } else {
          ridge_mlx
        }
        ridge_block <- ridge_val[block, , drop = FALSE]
        grad_block <- grad_block + ridge_block * beta[block, , drop = FALSE]
      } else {
        grad <- compute_grad(beta)
        grad_block <- grad[block, , drop = FALSE]
      }

      L_block <- lipschitz_mlx[block, , drop = FALSE]
      beta_block <- beta[block, , drop = FALSE]

      z_block <- beta_block - grad_block / L_block
      thresh_block <- lambda_mlx / L_block
      abs_z <- abs(z_block)
      magnitude <- mlx_maximum(abs_z - thresh_block, 0)
      beta_block_new <- sign(z_block) * magnitude

      start_row <- block[1]
      stop_row <- utils::tail(block, 1L)
      beta <- mlx_slice_update(beta, beta_block_new,
                               start = c(start_row, 1L),
                               stop = c(stop_row, dim(beta)[2]))

      if (!is.null(grad_cache) && grad_cache$type %in% c("gaussian", "binomial")) {
        delta <- beta_block_new - beta_block
        if (identical(grad_cache$type, "gaussian")) {
          grad_cache$residual <- grad_cache$residual - x_block %*% delta
        } else {
          grad_cache$eta <- grad_cache$eta + x_block %*% delta
          grad_cache$mu <- 1 / (1 + exp(-grad_cache$eta))
          grad_cache$residual <- grad_cache$mu - grad_cache$y
        }
      }
    }

    delta <- abs(beta - beta_prev)
    delta_max <- as.numeric(max(delta))
    if (!is.finite(delta_max)) {
      warning("Encountered non-finite updates during coordinate descent.", call. = FALSE)
      break
    }
    beta_vals <- as.numeric(beta)
    beta_finite <- all(is.finite(beta_vals))
    if (!beta_finite) {
      warning("Encountered non-finite coefficients during coordinate descent.", call. = FALSE)
      break
    }
    if (delta_max < tol) {
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
