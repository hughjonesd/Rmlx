test_that("mlx_grad computes gradients", {
  f <- function(x) {
    sum(x * x)
  }

  mat <- matrix(1:4, 2, 2)
  grads <- mlx_grad(f, mat)

  expect_length(grads, 1)
  expect_s3_class(grads[[1]], "mlx")
  expect_equal(as.matrix(grads[[1]]), 2 * mat, tolerance = 1e-6)
})

test_that("mlx_value_grad returns value and gradients", {
  f <- function(x, y) {
    sum((x - y) * (x - y))
  }

  x <- as_mlx(matrix(1:4, 2, 2))
  y <- as_mlx(matrix(4:1, 2, 2))

  res <- mlx_value_grad(f, x, y, argnums = c(1, 2))

  expect_s3_class(res$value, "mlx")
  expect_equal(length(res$grads), 2L)
  expect_equal(as.matrix(res$grads[[1]]), 2 * (as.matrix(x) - as.matrix(y)), tolerance = 1e-6)
  expect_equal(as.matrix(res$grads[[2]]), -2 * (as.matrix(x) - as.matrix(y)), tolerance = 1e-6)
})

test_that("mlx_stop_gradient detaches gradients", {
  f <- function(x) {
    sum(mlx_stop_gradient(x) * x)
  }

  x <- as_mlx(matrix(runif(4), 2, 2))
  grad <- mlx_grad(f, x)[[1]]
  expect_equal(as.matrix(grad), as.matrix(x), tolerance = 1e-6)
})

test_that("mlx_grad errors when function returns non-mlx", {
  bad_fun <- function(x) {
    sum(as.matrix(x))
  }

  expect_error(
    mlx_grad(bad_fun, matrix(1:4, 2, 2)),
    "must return an `mlx` object"
  )
})

test_that("mlx_grad errors when mixing base R operations", {
  bad_fun <- function(x) {
    tmp <- as.matrix(x)
    tmp <- tmp * tmp
    as_mlx(tmp)
  }

  expect_error(
    mlx_grad(bad_fun, matrix(1:4, 2, 2)),
    "Ensure all differentiable computations use MLX operations"
  )
})

test_that("subsetting participates in autograd", {
  f <- function(x) {
    row <- x[1, ]
    sum(row * row)
  }

  mat <- matrix(1:6, 2, 3)
  grads <- mlx_grad(f, mat)[[1]]

  expected <- matrix(0, nrow = 2, ncol = 3)
  expected[1, ] <- 2 * mat[1, ]

  expect_equal(as.matrix(grads), expected, tolerance = 1e-6)
})
