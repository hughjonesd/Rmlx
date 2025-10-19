test_that("print.mlx works", {
  skip_if_not_installed("Rmlx")
  skip_on_cran()

  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  expect_output(print(x_mlx), "mlx array")
  expect_output(print(x_mlx), "dtype")
  expect_output(print(x_mlx), "device")
})

test_that("dim.mlx works", {
  skip_if_not_installed("Rmlx")
  skip_on_cran()

  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  expect_equal(dim(x_mlx), c(3L, 4L))
})

test_that("length.mlx works", {
  skip_if_not_installed("Rmlx")
  skip_on_cran()

  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  expect_equal(length(x_mlx), 12L)
})

test_that("mlx_dim works", {
  skip_if_not_installed("Rmlx")
  skip_on_cran()

  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  expect_equal(mlx_dim(x_mlx), c(3L, 4L))
})

test_that("mlx_dtype works", {
  skip_if_not_installed("Rmlx")
  skip_on_cran()

  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x, dtype = "float32")

  expect_equal(mlx_dtype(x_mlx), "float32")
})
