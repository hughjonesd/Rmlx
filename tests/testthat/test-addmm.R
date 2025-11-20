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
