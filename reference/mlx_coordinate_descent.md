# Coordinate Descent with L1 Regularization

Minimizes f(beta) + lambda \* \|\|beta\|\|\_1 using coordinate descent,
where f is a smooth differentiable loss function.

## Usage

``` r
mlx_coordinate_descent(
  loss_fn,
  beta_init,
  lambda = 0,
  grad_fn = NULL,
  lipschitz = NULL,
  batch_size = NULL,
  compile = FALSE,
  max_iter = 1000,
  tol = 1e-06
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

- grad_fn:

  Optional gradient function. If NULL, computed via mlx_grad(loss_fn).

- lipschitz:

  Optional Lipschitz constants for each coordinate (length p vector). If
  NULL, uses simple constant estimates.

- batch_size:

  Number of coordinates to update per iteration (default: adaptive based
  on p).

  - 1 = pure coordinate descent (sequential)

  - p = full batch (all coordinates updated together)

  - intermediate values = mini-batch coordinate descent

- compile:

  Whether to compile the update step (default FALSE, not yet
  implemented).

- max_iter:

  Maximum number of iterations (default 1000).

- tol:

  Convergence tolerance (default 1e-6).

## Value

List with:

- beta: Optimized parameter vector (MLX tensor)

- n_iter: Number of iterations performed

- converged: Whether convergence criterion was met

## Details

This function implements proximal coordinate descent for problems of the
form: min_beta f(beta) + lambda \* \|\|beta\|\|\_1

where f is smooth. At each iteration, coordinates are updated via the
proximal gradient step: z_j = beta_j - (1/L_j) \* grad_f(beta)\_j beta_j
= soft_threshold(z_j, lambda/L_j)

where L_j is a Lipschitz constant for coordinate j.

Batching updates multiple coordinates simultaneously, which can
significantly improve performance by reducing R-to-MLX call overhead.

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
  batch_size = 10
)
```
