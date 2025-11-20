test_that("mlx_coordinate_descent solves lasso regression", {
  set.seed(42)
  n <- 100
  p <- 20

  # Generate sparse data
  X <- as_mlx(matrix(rnorm(n * p), n, p))
  beta_true <- as_mlx(matrix(c(1, -1, 0.5, rep(0, p - 3)), ncol = 1))
  y <- X %*% beta_true + as_mlx(matrix(rnorm(n, sd = 0.1), ncol = 1))

  # Define loss function (least squares)
  loss_fn <- function(beta) {
    residual <- y - X %*% beta
    sum(residual^2) / (2 * n)
  }

  # Solve with coordinate descent
  beta_init <- mlx_zeros(c(p, 1))
  result <- mlx_coordinate_descent(
    loss_fn = loss_fn,
    beta_init = beta_init,
    lambda = 0.1,
    max_iter = 100,
    tol = 1e-6
  )

  # Check convergence
  expect_true(result$converged)
  expect_lt(result$n_iter, 100)

  # Check solution is sparse
  beta_hat <- as.numeric(result$beta)
  n_nonzero <- sum(abs(beta_hat) > 1e-6)
  expect_lt(n_nonzero, p)

  # Check it reduces the loss
  final_loss <- as.numeric(loss_fn(result$beta))
  initial_loss <- as.numeric(loss_fn(beta_init))
  expect_lt(final_loss, initial_loss)
})

test_that("mlx_coordinate_descent with custom gradient", {
  set.seed(456)
  n <- 50
  p <- 10

  X <- as_mlx(matrix(rnorm(n * p), n, p))
  y <- as_mlx(matrix(rnorm(n), ncol = 1))

  # Loss and gradient for least squares
  loss_fn <- function(beta) {
    residual <- y - X %*% beta
    sum(residual^2) / (2 * n)
  }

  grad_fn <- function(beta) {
    residual <- y - X %*% beta
    -t(X) %*% residual / n
  }

  beta_init <- mlx_zeros(c(p, 1))

  # With custom gradient
  result1 <- mlx_coordinate_descent(
    loss_fn = loss_fn,
    beta_init = beta_init,
    lambda = 0.1,
    grad_fn = grad_fn,
    max_iter = 50
  )

  # With automatic gradient
  result2 <- mlx_coordinate_descent(
    loss_fn = loss_fn,
    beta_init = beta_init,
    lambda = 0.1,
    grad_fn = NULL,
    max_iter = 50
  )

  # Both should converge to similar solutions
  expect_true(result1$converged)
  expect_true(result2$converged)

  # Solutions should be close
  diff <- max(abs(as.numeric(result1$beta) - as.numeric(result2$beta)))
  expect_lt(diff, 1e-4)
})

test_that("mlx_coordinate_descent validates inputs", {
  loss_fn <- function(beta) sum(beta^2)
  beta_init <- mlx_matrix(rep(0, 10), 10, 1)

  # Should work with valid inputs
  expect_no_error(
    mlx_coordinate_descent(loss_fn, beta_init, lambda = 0.1, max_iter = 10)
  )

  # Should accept non-MLX beta and coerce automatically
  expect_no_error(
    mlx_coordinate_descent(loss_fn, matrix(0, 10, 1), lambda = 0.1)
  )

  # Should error with wrong length lipschitz
  expect_error(
    mlx_coordinate_descent(
      loss_fn, beta_init, lambda = 0.1,
      lipschitz = c(1, 2)  # Should be length 10
    ),
    "length equal"
  )
})

test_that("mlx_coordinate_descent keeps coefficients finite when supplied lipschitz constants", {
  set.seed(654)
  n <- 300
  p <- 100

  X_r <- matrix(rnorm(n * p), n, p)
  beta_true <- numeric(p)
  beta_true[sample(p, 3)] <- rnorm(3, sd = 3)
  y_r <- drop(X_r %*% beta_true + rnorm(n))

  X <- as_mlx(X_r)
  y <- mlx_matrix(y_r, ncol = 1)

  loss_fn <- function(beta) {
    residual <- y - X %*% beta
    sum(residual^2) / (2 * n)
  }

  grad_fn <- function(beta) {
    residual <- y - X %*% beta
    -crossprod(X, residual) / n
  }

  beta_init <- mlx_zeros(c(p, 1))
  lipschitz <- colSums(X_r^2) / n + 1e-8

  result <- mlx_coordinate_descent(
    loss_fn = loss_fn,
    beta_init = beta_init,
    lambda = 0.01,
    grad_fn = grad_fn,
    lipschitz = lipschitz,
    max_iter = 1000,
    tol = 1e-6,
    block_size = 8
  )

  beta_vals <- as.numeric(as.matrix(result$beta))
  expect_true(result$converged,
              info = "With accurate Lipschitz constants this quadratic problem should converge.")
  expect_false(any(is.nan(beta_vals)),
               info = "Coordinate descent should not introduce NaNs when gradients are finite.")
})
