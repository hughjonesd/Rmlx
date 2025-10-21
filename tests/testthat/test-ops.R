test_that("arithmetic operations work", {
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
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  z <- as.matrix(-x_mlx)
  expect_equal(z, -x, tolerance = 1e-6)
})

test_that("comparison operators work", {
  x <- matrix(1:12, 3, 4)
  y <- matrix(c(1:6, 13:18), 3, 4)

  x_mlx <- as_mlx(x)
  y_mlx <- as_mlx(y)

  # Less than
  lt <- x_mlx < y_mlx
  z <- as.matrix(lt)
  expect_equal(z, x < y)
  expect_equal(lt$dtype, "bool")

  # Equal
  z <- as.matrix(x_mlx == y_mlx)
  expect_equal(z, x == y)

  # Greater than
  z <- as.matrix(x_mlx > y_mlx)
  expect_equal(z, x > y)
})

test_that("scalar operations work", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  # Scalar addition
  z <- as.matrix(x_mlx + 10)
  expect_equal(z, x + 10, tolerance = 1e-6)

  # Scalar multiplication
  z <- as.matrix(x_mlx * 2)
  expect_equal(z, x * 2, tolerance = 1e-6)
})

test_that("boolean operands coerce for arithmetic", {
  bool_mat <- matrix(c(TRUE, FALSE, TRUE, FALSE), 2, 2)
  num_mat <- matrix(1:4, 2, 2)

  bool_mlx <- as_mlx(bool_mat)
  num_mlx <- as_mlx(num_mat)

  sum_obj <- bool_mlx + num_mlx
  sum_res <- as.matrix(sum_obj)
  expect_equal(sum_res, num_mat + (bool_mat * 1), tolerance = 1e-6)
  expect_equal(sum_obj$dtype, "float32")

  bool_sum_obj <- bool_mlx + bool_mlx
  bool_sum <- as.matrix(bool_sum_obj)
  expect_equal(bool_sum, (bool_mat * 1) + (bool_mat * 1), tolerance = 1e-6)
  expect_equal(bool_sum_obj$dtype, "float32")
})

test_that("binary operations align devices and dtypes", {
  old_device <- mlx_default_device()
  on.exit(mlx_default_device(old_device))

  mlx_default_device("gpu")

  x_gpu <- as_mlx(matrix(1:4, 2, 2), device = "gpu", dtype = "float32")
  y_cpu <- as_mlx(matrix(5:8, 2, 2), device = "cpu")

  result <- x_gpu + y_cpu

  expect_equal(result$device, "gpu")
  expect_equal(result$dtype, "float32")
  expect_equal(as.matrix(result), matrix(c(6, 8, 10, 12), 2, 2), tolerance = 1e-6)
})
