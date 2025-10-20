test_that("matrix multiplication works", {
  a <- matrix(1:6, 2, 3)
  b <- matrix(1:6, 3, 2)

  a_mlx <- as_mlx(a)
  b_mlx <- as_mlx(b)

  c_mlx <- a_mlx %*% b_mlx
  c <- as.matrix(c_mlx)

  expect_equal(c, a %*% b, tolerance = 1e-6)
})

test_that("matrix multiplication aligns devices and dtypes", {
  old_device <- mlx_default_device()
  on.exit(mlx_default_device(old_device))

  mlx_default_device("gpu")

  a <- matrix(1:6, 2, 3)
  b <- matrix(7:12, 3, 2)

  a_gpu <- as_mlx(a, device = "gpu", dtype = "float32")
  b_cpu <- as_mlx(b, device = "cpu")

  c_mlx <- a_gpu %*% b_cpu

  expect_equal(c_mlx$device, "gpu")
  expect_equal(c_mlx$dtype, "float32")
  expect_equal(as.matrix(c_mlx), a %*% b, tolerance = 1e-5)
})

test_that("matrix multiplication dimension checking works", {
  a <- matrix(1:6, 2, 3)
  b <- matrix(1:6, 2, 3)  # Non-conformable

  a_mlx <- as_mlx(a)
  b_mlx <- as_mlx(b)

  expect_error(a_mlx %*% b_mlx, "Non-conformable")
})

test_that("transpose works", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  xt_mlx <- t(x_mlx)
  xt <- as.matrix(xt_mlx)

  expect_equal(xt, t(x), tolerance = 1e-6)
  expect_equal(xt_mlx$dim, c(4L, 3L))
})

test_that("crossprod works", {
  x <- matrix(rnorm(20), 5, 4)
  x_mlx <- as_mlx(x)

  # crossprod(x)
  xtx <- as.matrix(crossprod(x_mlx))
  expect_equal(xtx, crossprod(x), tolerance = 1e-6)

  # crossprod(x, y)
  y <- matrix(rnorm(20), 5, 4)
  y_mlx <- as_mlx(y)
  xty <- as.matrix(crossprod(x_mlx, y_mlx))
  expect_equal(xty, crossprod(x, y), tolerance = 1e-6)
})

test_that("tcrossprod works", {
  x <- matrix(rnorm(20), 5, 4)
  x_mlx <- as_mlx(x)

  # tcrossprod(x)
  xxt <- as.matrix(tcrossprod(x_mlx))
  expect_equal(xxt, tcrossprod(x), tolerance = 1e-6)

  # tcrossprod(x, y)
  y <- matrix(rnorm(20), 5, 4)
  y_mlx <- as_mlx(y)
  xyt <- as.matrix(tcrossprod(x_mlx, y_mlx))
  expect_equal(xyt, tcrossprod(x, y), tolerance = 1e-6)
})
