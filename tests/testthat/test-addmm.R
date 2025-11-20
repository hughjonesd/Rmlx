test_that("mlx_addmm matches manual add + matmul", {
  set.seed(123)
  mat1 <- matrix(rnorm(6), 2, 3)
  mat2 <- matrix(rnorm(9), 3, 3)
  input <- matrix(rnorm(6), 2, 3)

  res <- mlx_addmm(input, mat1, mat2)
  manual <- input + mat1 %*% mat2

  expect_s3_class(res, "mlx")
  expect_equal(as.matrix(res), manual, tolerance = 1e-6)
})

test_that("mlx_addmm respects alpha and beta scaling", {
  set.seed(456)
  mat1 <- matrix(rnorm(4), 2, 2)
  mat2 <- matrix(rnorm(4), 2, 2)
  input <- matrix(rnorm(4), 2, 2)

  alpha <- 0.5
  beta <- 2

  res <- mlx_addmm(input, mat1, mat2, alpha = alpha, beta = beta)
  manual <- beta * input + alpha * (mat1 %*% mat2)

  expect_equal(as.matrix(res), manual, tolerance = 1e-6)
})

test_that("mlx_addmm promotes dtype and device", {
  mat1 <- mlx_matrix(1:4, 2, 2, dtype = "float32", device = "gpu")
  mat2 <- mlx_matrix(5:8, 2, 2, dtype = "float32", device = "gpu")
  input <- mlx_matrix(rep(1, 4), 2, 2, dtype = "float32", device = "gpu")

  res <- mlx_addmm(input, mat1, mat2)

  expect_identical(mlx_dtype(res), "float32")
  expect_identical(res$device, "gpu")
  expect_equal(as.matrix(res), as.matrix(input) + as.matrix(mat1) %*% as.matrix(mat2), tolerance = 1e-6)
})

test_that("mlx_addmm errors on non-conformable mat1 and mat2", {
  # mat1 is 2x3, mat2 is 4x2 - inner dimensions don't match
  mat1 <- mlx_matrix(1:6, 2, 3)
  mat2 <- mlx_matrix(1:8, 4, 2)
  input <- mlx_matrix(1:4, 2, 2)

  expect_error(
    mlx_addmm(input, mat1, mat2),
    "Non-conformable operands: mat1 is 2 x 3 but mat2 is 4 x 2"
  )
})

test_that("mlx_addmm errors when mat1 columns don't match mat2 rows", {
  # mat1 is 3x2, mat2 is 3x4 - can't multiply (need mat1[,2] == mat2[1,])
  mat1 <- mlx_matrix(1:6, 3, 2)
  mat2 <- mlx_matrix(1:12, 3, 4)
  input <- mlx_matrix(1:12, 3, 4)

  expect_error(
    mlx_addmm(input, mat1, mat2),
    "Non-conformable operands: mat1 is 3 x 2 but mat2 is 3 x 4"
  )
})

test_that("mlx_addmm errors when input shape doesn't match result", {
  # mat1 %*% mat2 will be 2x3, but input is 2x2
  mat1 <- mlx_matrix(1:4, 2, 2)
  mat2 <- mlx_matrix(1:6, 2, 3)
  input <- mlx_matrix(1:4, 2, 2)

  expect_error(
    mlx_addmm(input, mat1, mat2),
    "Input shape \\(2 x 2\\) must match mat1 %\\*% mat2 result \\(2 x 3\\)"
  )
})

test_that("mlx_addmm errors when input has wrong dimensions", {
  # mat1 %*% mat2 will be 3x3, but input is 3x2
  mat1 <- mlx_matrix(1:9, 3, 3)
  mat2 <- mlx_matrix(1:9, 3, 3)
  input <- mlx_matrix(1:6, 3, 2)

  expect_error(
    mlx_addmm(input, mat1, mat2),
    "Input shape \\(3 x 2\\) must match mat1 %\\*% mat2 result \\(3 x 3\\)"
  )
})

test_that("mlx_addmm errors on non-2D mat1", {
  mat1 <- as_mlx(1:6)  # 1D vector
  mat2 <- mlx_matrix(1:6, 2, 3)
  input <- mlx_matrix(1:6, 2, 3)

  expect_error(
    mlx_addmm(input, mat1, mat2),
    "mlx_addmm requires mat1 and mat2 to be 2D matrices"
  )
})

test_that("mlx_addmm errors on non-2D mat2", {
  mat1 <- mlx_matrix(1:6, 2, 3)
  mat2 <- as_mlx(1:6)  # 1D vector
  input <- mlx_matrix(1:4, 2, 2)

  expect_error(
    mlx_addmm(input, mat1, mat2),
    "mlx_addmm requires mat1 and mat2 to be 2D matrices"
  )
})

test_that("mlx_addmm errors on non-2D input", {
  mat1 <- mlx_matrix(1:6, 2, 3)
  mat2 <- mlx_matrix(1:6, 3, 2)
  input <- as_mlx(1:4)  # 1D vector

  expect_error(
    mlx_addmm(input, mat1, mat2),
    "mlx_addmm requires input to be a 2D matrix"
  )
})

test_that("mlx_addmm errors on 3D array inputs", {
  mat1 <- mlx_array(1:24, c(2, 3, 4))  # 3D array
  mat2 <- mlx_matrix(1:6, 3, 2)
  input <- mlx_matrix(1:4, 2, 2)

  expect_error(
    mlx_addmm(input, mat1, mat2),
    "mlx_addmm requires mat1 and mat2 to be 2D matrices"
  )
})
