test_that("arithmetic operations work", {
  skip_if_not_installed("Rmlx")
  skip_on_cran()

  x <- matrix(1:12, 3, 4)
  y <- matrix(13:24, 3, 4)

  x_mlx <- as_mlx(x)
  y_mlx <- as_mlx(y)

  # Addition
  z <- as.matrix(x_mlx + y_mlx)
  expect_equal(z, x + y, tolerance = 1e-6)

  # Subtraction
  z <- as.matrix(x_mlx - y_mlx)
  expect_equal(z, x - y, tolerance = 1e-6)

  # Multiplication
  z <- as.matrix(x_mlx * y_mlx)
  expect_equal(z, x * y, tolerance = 1e-6)

  # Division
  z <- as.matrix(x_mlx / y_mlx)
  expect_equal(z, x / y, tolerance = 1e-6)

  # Power
  z <- as.matrix(x_mlx ^ 2)
  expect_equal(z, x ^ 2, tolerance = 1e-6)
})

test_that("unary negation works", {
  skip_if_not_installed("Rmlx")
  skip_on_cran()

  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  z <- as.matrix(-x_mlx)
  expect_equal(z, -x, tolerance = 1e-6)
})

test_that("comparison operators work", {
  skip_if_not_installed("Rmlx")
  skip_on_cran()

  x <- matrix(1:12, 3, 4)
  y <- matrix(c(1:6, 13:18), 3, 4)

  x_mlx <- as_mlx(x)
  y_mlx <- as_mlx(y)

  # Less than
  z <- as.matrix(x_mlx < y_mlx)
  expect_equal(z, x < y)

  # Equal
  z <- as.matrix(x_mlx == y_mlx)
  expect_equal(z, x == y)

  # Greater than
  z <- as.matrix(x_mlx > y_mlx)
  expect_equal(z, x > y)
})

test_that("scalar operations work", {
  skip_if_not_installed("Rmlx")
  skip_on_cran()

  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  # Scalar addition
  z <- as.matrix(x_mlx + 10)
  expect_equal(z, x + 10, tolerance = 1e-6)

  # Scalar multiplication
  z <- as.matrix(x_mlx * 2)
  expect_equal(z, x * 2, tolerance = 1e-6)
})
