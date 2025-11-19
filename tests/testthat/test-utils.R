test_that("print.mlx works", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  expect_output(print(x_mlx), "mlx array")
  expect_output(print(x_mlx), "dtype")
  expect_output(print(x_mlx), "device")
})

test_that("dim.mlx works", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  expect_equal(dim(x_mlx), c(3L, 4L))
})

test_that("length.mlx works", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  expect_equal(length(x_mlx), 12L)
})

test_that("mlx_shape works", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  expect_equal(mlx_shape(x_mlx), c(3L, 4L))
})

test_that("mlx_dtype works", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x, dtype = "float32")

  expect_equal(mlx_dtype(x_mlx), "float32")
})

test_that("dim<-.mlx works for basic reshaping", {
  x <- as_mlx(1:12)
  expect_null(dim(x))
  expect_equal(mlx_shape(x), 12L)

  dim(x) <- c(3, 4)
  expect_equal(dim(x), c(3L, 4L))
  expect_equal(as.matrix(x), matrix(1:12, 3, 4, byrow = TRUE))

  # Reshape to different dimensions
  dim(x) <- c(2, 6)
  expect_equal(dim(x), c(2L, 6L))
  expect_equal(length(x), 12L)
})

test_that("dim<-.mlx works for 3D arrays", {
  x <- as_mlx(1:24)
  dim(x) <- c(2, 3, 4)

  expect_equal(dim(x), c(2L, 3L, 4L))
  expect_equal(length(x), 24L)
})

test_that("dim<-.mlx preserves device and dtype", {
  x <- as_mlx(1:12, device = "gpu", dtype = "float32")
  dim(x) <- c(3, 4)

  expect_equal(x$device, "gpu")
  expect_equal(mlx_dtype(x), "float32")
})

test_that("dim<-.mlx errors when product doesn't match", {
  x <- as_mlx(1:12)

  expect_error(
    dim(x) <- c(3, 5),
    "dims \\[product 15\\] do not match the length of object \\[12\\]"
  )

  expect_error(
    dim(x) <- c(2, 7),
    "dims \\[product 14\\] do not match the length of object \\[12\\]"
  )
})

test_that("dim<-.mlx errors on invalid dimensions", {
  x <- as_mlx(1:12)

  expect_error(
    dim(x) <- c(-1, 12),
    "dims cannot be negative"
  )

  expect_error(
    dim(x) <- c(3, -4),
    "dims cannot be negative"
  )

  expect_error(
    dim(x) <- NA,
    "dims must be a numeric vector without NAs"
  )

  expect_error(
    dim(x) <- c(3, NA),
    "dims must be a numeric vector without NAs"
  )
})

test_that("dim<-.mlx handles edge cases", {
  # Single element
  x <- as_mlx(5)
  dim(x) <- 1
  expect_null(dim(x))
  expect_equal(mlx_shape(x), 1L)
  expect_equal(as.numeric(x), 5)

  # Reshape 1D to 1D
  x <- as_mlx(1:10)
  dim(x) <- 10
  expect_null(dim(x))
  expect_equal(mlx_shape(x), 10L)
})
