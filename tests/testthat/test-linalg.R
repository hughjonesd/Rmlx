test_that("solve works for linear systems", {
  # Test solving Ax = b
  set.seed(123)
  A <- matrix(rnorm(9), 3, 3)
  b <- matrix(rnorm(3), 3, 1)

  A_mlx <- as_mlx(A)
  b_mlx <- as_mlx(b)

  x_mlx <- solve(A_mlx, b_mlx)
  x_r <- solve(A, b)

  expect_equal(as.matrix(x_mlx), x_r, tolerance = 1e-5)

  # Verify solution: A %*% x should equal b
  result_mlx <- A_mlx %*% x_mlx
  expect_equal(as.matrix(result_mlx), b, tolerance = 1e-5)
})

test_that("solve works for matrix inverse", {
  # Test computing inverse
  set.seed(456)
  A <- matrix(rnorm(16), 4, 4)

  A_mlx <- as_mlx(A)

  A_inv_mlx <- solve(A_mlx)
  A_inv_r <- solve(A)

  expect_equal(as.matrix(A_inv_mlx), A_inv_r, tolerance = 1e-5)

  # Verify: A %*% A_inv should be identity
  I_mlx <- A_mlx %*% A_inv_mlx
  I_expected <- diag(4)
  expect_equal(as.matrix(I_mlx), I_expected, tolerance = 1e-5)
})

test_that("solve works with vector b", {
  # Test with b as a vector (not matrix)
  set.seed(789)
  A <- matrix(rnorm(9), 3, 3)
  b <- rnorm(3)

  A_mlx <- as_mlx(A)
  b_mlx <- as_mlx(b)

  x_mlx <- solve(A_mlx, b_mlx)
  x_r <- solve(A, b)

  expect_equal(as.numeric(as.matrix(x_mlx)), x_r, tolerance = 1e-5)
})

test_that("solve works when A is mlx and b is R matrix", {
  # Test automatic conversion of b from R to mlx
  set.seed(321)
  A <- matrix(rnorm(9), 3, 3)
  b <- matrix(rnorm(3), 3, 1)

  A_mlx <- as_mlx(A)
  # b is NOT converted to mlx - should be auto-converted

  x_mlx <- solve(A_mlx, b)
  x_r <- solve(A, b)

  expect_equal(as.matrix(x_mlx), x_r, tolerance = 1e-5)
})

test_that("solve stages to cpu and restores gpu device", {
  old_device <- mlx_default_device()
  on.exit(mlx_default_device(old_device))

  mlx_default_device("gpu")

  set.seed(987)
  A <- matrix(rnorm(9), 3, 3)
  b <- matrix(rnorm(3), 3, 1)

  A_gpu <- as_mlx(A, device = "gpu", dtype = "float32")
  b_gpu <- as_mlx(b, device = "gpu", dtype = "float32")

  x_gpu <- solve(A_gpu, b_gpu)

  expect_equal(x_gpu$device, "gpu")
  expect_equal(x_gpu$dtype, "float32")
  expect_equal(as.matrix(x_gpu), solve(A, b), tolerance = 1e-5)

  A_inv_gpu <- solve(A_gpu)
  expect_equal(A_inv_gpu$device, "gpu")
  expect_equal(as.matrix(A_inv_gpu), solve(A), tolerance = 1e-5)
})

test_that("cholesky matches base R", {
  set.seed(123)
  A <- matrix(rnorm(9), 3, 3)
  spd <- crossprod(A) + diag(3) * 1e-3

  chol_r <- chol(spd)
  chol_mlx <- chol(as_mlx(spd))

  expect_equal(as.matrix(chol_mlx), chol_r, tolerance = 1e-5)
})

test_that("qr decomposition reconstructs the original matrix", {
  set.seed(42)
  A <- matrix(rnorm(12), 4, 3)
  qr_mlx <- qr(as_mlx(A))

  Q <- as.matrix(qr_mlx$Q)
  R <- as.matrix(qr_mlx$R)

  expect_equal(Q %*% R, A, tolerance = 1e-5)
  expect_equal(t(Q) %*% Q, diag(3), tolerance = 1e-5)
})

test_that("svd reconstructs the original matrix", {
  set.seed(101)
  A <- matrix(rnorm(12), 3, 4)
  svd_mlx <- svd(as_mlx(A))

  U <- as.matrix(svd_mlx$u)
  d <- svd_mlx$d
  V <- as.matrix(svd_mlx$v)

  reconstructed <- U %*% diag(d) %*% t(V)
  expect_equal(reconstructed, A, tolerance = 1e-4)
})

test_that("svd.mlx with nu=0 and nv=0 returns only singular values", {
  set.seed(404)
  A <- matrix(rnorm(12), 3, 4)
  svd_mlx <- svd(as_mlx(A), nu = 0, nv = 0)

  expect_null(svd_mlx$u)
  expect_true(is.numeric(svd_mlx$d))
  expect_null(svd_mlx$v)
  expect_equal(length(svd_mlx$d), 3)
})

test_that("pinv matches analytical pseudoinverse for full-rank matrix", {
  set.seed(202)
  A <- matrix(rnorm(12), 4, 3)
  pseudo_r <- solve(t(A) %*% A) %*% t(A)

  pseudo_mlx <- pinv(as_mlx(A))
  expect_equal(as.matrix(pseudo_mlx), pseudo_r, tolerance = 1e-4)
})
