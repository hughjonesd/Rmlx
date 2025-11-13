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
    batch_size = 5,
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

test_that("mlx_coordinate_descent works with different batch sizes", {
  set.seed(123)
  n <- 50
  p <- 10

  X <- as_mlx(matrix(rnorm(n * p), n, p))
  y <- as_mlx(matrix(rnorm(n), ncol = 1))

  loss_fn <- function(beta) {
    residual <- y - X %*% beta
    sum(residual^2) / (2 * n)
  }

  beta_init <- mlx_zeros(c(p, 1))

  # Test different batch sizes
  for (bs in c(1, 5, 10)) {
    result <- mlx_coordinate_descent(
      loss_fn = loss_fn,
      beta_init = beta_init,
      lambda = 0.05,
      batch_size = bs,
      max_iter = 50
    )

    expect_true(result$converged || result$n_iter == 50)
    expect_true(is.mlx(result$beta))
    expect_equal(dim(result$beta), c(p, 1))
  }
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
    batch_size = 5,
    max_iter = 50
  )

  # With automatic gradient
  result2 <- mlx_coordinate_descent(
    loss_fn = loss_fn,
    beta_init = beta_init,
    lambda = 0.1,
    grad_fn = NULL,
    batch_size = 5,
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
  beta_init <- as_mlx(matrix(0, 10, 1))

  # Should work with valid inputs
  expect_no_error(
    mlx_coordinate_descent(loss_fn, beta_init, lambda = 0.1, max_iter = 10)
  )

  # Should error with non-MLX beta
  expect_error(
    mlx_coordinate_descent(loss_fn, matrix(0, 10, 1), lambda = 0.1),
    "MLX array"
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
