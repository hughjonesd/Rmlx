# Linear Regression with MLX

## Overview

This vignette demonstrates linear regression using Rmlx, based on the
[MLX linear regression
example](https://ml-explore.github.io/mlx/build/html/examples/linear_regression.html).
We’ll train a linear model using automatic differentiation and
stochastic gradient descent (SGD) on GPU-accelerated arrays.

## Problem Setup

We’ll create:

- A random “true” weight vector `w_star`
- A random design matrix `X`
- Noisy labels `y = X %*% w_star + small_noise`

``` r
library(Rmlx)
#> 
#> Attaching package: 'Rmlx'
#> The following object is masked from 'package:stats':
#> 
#>     fft
#> The following objects are masked from 'package:base':
#> 
#>     asplit, backsolve, chol2inv, col, colMeans, colSums, diag, drop,
#>     outer, row, rowMeans, rowSums, svd

# Problem metadata
num_features <- 100
num_cases <- 10000
num_iters <- 1200          # iterations of SGD
learning_rate <- 0.01      # learning rate for SGD

# Set seed for reproducibility
set.seed(42)

# True parameters (what we're trying to learn)
w_star <- mlx_rand_normal(c(num_features, 1))

# Input examples (design matrix)
X <- mlx_rand_normal(c(num_cases, num_features))

# Noisy labels
eps <- mlx_rand_normal(c(num_cases, 1))
y <- X %*% w_star + eps
```

## Define the Loss Function

The mean squared error loss is a standard choice for regression:

``` r
# Define loss function
loss_fn <- function(w) {
  preds <- X %*% w
  residuals <- preds - y
  0.5 * mean(residuals * residuals)
}
```

The loss measures how well our parameters `w` predict the labels. Lower
loss means better predictions.

## Automatic Differentiation

Rmlx provides
[`mlx_grad()`](https://hughjonesd.github.io/Rmlx/reference/mlx_grad.md)
to compute gradients via automatic differentiation. This computes the
gradient of the loss with respect to our parameters:

``` r
# Get the gradient function
grad_fn <- function(w) {
  mlx_grad(loss_fn, w)[[1]]
}

train_sgd <- function(steps = num_iters, step_size = learning_rate, verbose = TRUE) {
  w <- 1e-2 * mlx_rand_normal(c(num_features, 1))
  for (i in seq_len(steps)) {
    grad <- grad_fn(w)
    w <- w - step_size * grad
    mlx_eval(w)
    if (verbose && i %% 1000 == 0) {
      cat("Iteration", i, "- Loss:", as.vector(loss_fn(w)), "\n")
    }
  }
  w
}
```

## Training Loop with SGD

We train by repeatedly computing gradients and updating parameters. In
each iteration, we:

1.  Compute the gradient of loss with respect to `w`
2.  Update parameters using the gradient step
3.  Force evaluation to prevent the computation graph from growing
    unbounded
4.  Monitor progress by printing loss every 1000 iterations

``` r
w_sgd <- train_sgd()
#> Iteration 1000 - Loss: 0.4926799
```

## Method 2: Closed-form Regression via Matrix Algebra

Gradient descent is flexible, but linear regression also has a
closed-form solution that can be obtained via the QR decomposition.
Rather than forming $X^{\top}X$ explicitly, we factor $X = QR$ with
$Q^{\top}Q = I$ and solve the triangular system $Rw = Q^{\top}y$:

``` r
mlx_normal_eq <- function(X, y) {
  qr_res <- qr(X)
  q <- qr_res$Q
  r <- qr_res$R
  q_ty <- crossprod(q, y)
  mlx_solve_triangular(r, q_ty, upper = TRUE)
}

w_closed <- mlx_normal_eq(X, y)
mlx_eval(w_closed)

closed_error <- w_closed - w_star
closed_error_norm <- sqrt(sum(closed_error * closed_error))
cat("Closed-form ||w - w*|| =", as.vector(closed_error_norm), "\n")
#> Closed-form ||w - w*|| = 0.1064947
```

## Accelerating the Closed-form Solution with `mlx_compile()`

The closed-form function mixes several MLX primitives. We can trace and
fuse those operations with
[`mlx_compile()`](https://hughjonesd.github.io/Rmlx/reference/mlx_compile.md).
The first call incurs the tracing cost; subsequent calls reuse the
compiled graph.

``` r
compiled_normal_eq <- mlx_compile(mlx_normal_eq)

# Warm-up call performs tracing and compilation
mlx_eval(compiled_normal_eq(X, y))

# Re-use the compiled function
w_compiled <- compiled_normal_eq(X, y)
mlx_eval(w_compiled)

compiled_error <- w_compiled - w_star
compiled_error_norm <- sqrt(sum(compiled_error * compiled_error))
cat("Compiled closed-form ||w - w*|| =", as.vector(compiled_error_norm), "\n")
#> Compiled closed-form ||w - w*|| = 0.1064947
```

## Accuracy and Performance Comparison

To compare approaches we measure elapsed time over several repetitions
and the resulting distance between each estimate and the true
coefficients. We also add base R’s normal-equation implementation as a
reference.

``` r
library(bench)

# Fit models once for accuracy measurements
w_sgd <- train_sgd(verbose = FALSE)
w_closed <- mlx_normal_eq(X, y)
compiled_normal_eq <- mlx_compile(mlx_normal_eq)
mlx_eval(compiled_normal_eq(X, y))
w_compiled <- compiled_normal_eq(X, y)
X_r <- as.matrix(X)
y_r <- as.matrix(y)
w_base <- matrix(lm.fit(X_r, y_r[, 1])$coefficients, ncol = 1)

# Accuracy comparisons
to_norm <- function(w_hat) {
  diff <- w_hat - w_star
  rss <- sqrt(sum(diff * diff))
  as.vector(rss)
}

# Benchmark timings (compiled solution already warm)
timings <- bench::mark(
  sgd = {
    res <- train_sgd(verbose = FALSE)
    mlx_eval(res)
  },
  mlx_closed = {
    res <- mlx_normal_eq(X, y)
    mlx_eval(res)
  },
  mlx_closed_compiled = {
    res <- compiled_normal_eq(X, y)
    mlx_eval(res)
  },
  base_R = {
    lm.fit(X_r, y_r[, 1])$coefficients
  },
  check = FALSE
) |>
  as.data.frame()
#> Warning: Some expressions had a GC in every iteration; so filtering is
#> disabled.

results <- data.frame(
  method = c("SGD", "MLX closed form", "MLX closed form (compiled)", "Base R"),
  median_time = timings$median,
  parameter_error = c(
    to_norm(w_sgd),
    to_norm(w_closed),
    to_norm(w_compiled),
    to_norm(w_base)
  )
)
knitr::kable(results, digits = 4)
```

| method                     | median_time | parameter_error |
|:---------------------------|------------:|----------------:|
| SGD                        |       1.98s |          0.1065 |
| MLX closed form            |     34.57ms |          0.1065 |
| MLX closed form (compiled) |     24.68ms |          0.1065 |
| Base R                     |     39.23ms |          0.1065 |

## Device Selection

By default, computations run on the best available device. Switch to CPU
if needed:

``` r
# Use CPU (useful for debugging)
mlx_default_device("cpu")

with_default_device("cpu", {
  ...
})
```
