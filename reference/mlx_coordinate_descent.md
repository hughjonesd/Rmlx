# Coordinate Descent with L1 Regularization

Minimizes f(beta) + lambda \* \|\|beta\|\|\_1 using coordinate descent,
where f is a smooth differentiable loss function.

## Usage

``` r
mlx_coordinate_descent(
  loss_fn,
  beta_init,
  lambda = 0,
  ridge_penalty = 0,
  grad_fn = NULL,
  lipschitz = NULL,
  max_iter = 1000,
  tol = 1e-06,
  block_size = 1,
  grad_cache = NULL
)
```

## Arguments

- loss_fn:

  Function(beta) -\> scalar loss (MLX tensor). Must be smooth and
  differentiable.

- beta_init:

  Initial parameter vector (p x 1 MLX tensor).

- lambda:

  L1 penalty parameter (scalar, default 0).

- ridge_penalty:

  Optional ridge (L2) penalty term applied per-coordinate when computing
  gradients. Can be a scalar, numeric vector of length p, or an `mlx`
  array with shape compatible with `beta_init`.

- grad_fn:

  Optional gradient function. If NULL, computed via mlx_grad(loss_fn).

- lipschitz:

  Optional Lipschitz constants for each coordinate (length p vector). If
  NULL, uses simple constant estimates.

- max_iter:

  Maximum number of iterations (default 1000).

- tol:

  Convergence tolerance (default 1e-6).

- block_size:

  Number of coordinates to update before recomputing the gradient. Set
  to 1 for strict coordinate descent; larger values trade accuracy for
  speed.

- grad_cache:

  Optional environment for supplying cached gradient components.
  Supported fields are `type = "gaussian"` with entries `x`, `residual`,
  `n_obs`, and optional `ridge_penalty`; or `type = "binomial"` with
  entries `x`, `eta`, `mu`, `residual`, `y`, `n_obs`, and optional
  `ridge_penalty`.

## Value

List with:

- beta: Optimized parameter vector (MLX tensor)

- n_iter: Number of iterations performed

- converged: Whether convergence criterion was met

## Details

This function implements proximal gradient descent for problems of the
form: min_beta f(beta) + lambda \* \|\|beta\|\|\_1

where f is smooth. At each iteration, all coordinates are updated via: z
= beta - (1/L) \* grad_f(beta) beta = soft_threshold(z, lambda/L)

where L are Lipschitz constants for each coordinate.

## Examples

``` r
# Lasso regression: min 0.5*||y - X*beta||^2 + lambda*||beta||_1
n <- 100
p <- 50
X <- as_mlx(matrix(rnorm(n*p), n, p))
y <- as_mlx(matrix(rnorm(n), ncol=1))
beta_init <- mlx_zeros(c(p, 1))

loss_fn <- function(beta) {
  residual <- y - X %*% beta
  sum(residual^2) / (2*n)
}

result <- mlx_coordinate_descent(
  loss_fn = loss_fn,
  beta_init = beta_init,
  lambda = 0.1,
  block_size = 8
)

# Reuse cached residuals for a Gaussian problem
grad_cache <- new.env(parent = emptyenv())
grad_cache$type <- "gaussian"
grad_cache$x <- X
grad_cache$n_obs <- n
grad_cache$residual <- y - X %*% beta_init
cached <- mlx_coordinate_descent(
  loss_fn = loss_fn,
  beta_init = beta_init,
  lambda = 0.1,
  grad_cache = grad_cache
)
```
