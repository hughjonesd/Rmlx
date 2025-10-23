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

test_that("chol.mlx works with different matrix sizes", {
  set.seed(234)

  # 2x2 matrix
  A2 <- matrix(rnorm(4), 2, 2)
  spd2 <- crossprod(A2) + diag(2) * 0.1
  expect_equal(as.matrix(chol(as_mlx(spd2))), chol(spd2), tolerance = 1e-5)

  # 5x5 matrix
  A5 <- matrix(rnorm(25), 5, 5)
  spd5 <- crossprod(A5) + diag(5) * 0.1
  expect_equal(as.matrix(chol(as_mlx(spd5))), chol(spd5), tolerance = 1e-5)
})

test_that("chol.mlx reconstruction works", {
  set.seed(345)
  A <- matrix(rnorm(16), 4, 4)
  spd <- crossprod(A) + diag(4) * 0.5

  R <- chol(as_mlx(spd))
  reconstructed <- t(R) %*% R

  expect_equal(as.matrix(reconstructed), spd, tolerance = 1e-5)
})

test_that("chol.mlx preserves device and dtype", {
  old_device <- mlx_default_device()
  on.exit(mlx_default_device(old_device))

  mlx_default_device("gpu")

  set.seed(456)
  A <- matrix(rnorm(9), 3, 3)
  spd <- crossprod(A) + diag(3) * 0.1

  spd_gpu <- as_mlx(spd, device = "gpu", dtype = "float32")
  R_gpu <- chol(spd_gpu)

  expect_equal(R_gpu$device, "gpu")
  expect_equal(R_gpu$dtype, "float32")
  expect_equal(as.matrix(R_gpu), chol(spd), tolerance = 1e-4)
})

test_that("chol.mlx errors with pivot = TRUE", {
  set.seed(567)
  A <- matrix(rnorm(9), 3, 3)
  spd <- crossprod(A) + diag(3) * 0.1

  expect_error(chol(as_mlx(spd), pivot = TRUE), "pivoted Cholesky is not supported")
})

test_that("chol.mlx errors with LINPACK = TRUE", {
  set.seed(678)
  A <- matrix(rnorm(9), 3, 3)
  spd <- crossprod(A) + diag(3) * 0.1

  expect_error(chol(as_mlx(spd), LINPACK = TRUE), "LINPACK routines are not supported")
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

test_that("svd.mlx singular values match base R", {
  set.seed(505)
  A <- matrix(rnorm(20), 4, 5)

  svd_r <- svd(A)
  svd_mlx <- svd(as_mlx(A))

  expect_equal(svd_mlx$d, svd_r$d, tolerance = 1e-5)
})

test_that("svd.mlx works with different matrix dimensions", {
  set.seed(606)

  # Tall matrix (more rows than columns)
  A_tall <- matrix(rnorm(15), 5, 3)
  svd_tall <- svd(as_mlx(A_tall))
  U <- as.matrix(svd_tall$u)
  V <- as.matrix(svd_tall$v)
  reconstructed_tall <- U %*% diag(svd_tall$d) %*% t(V)
  expect_equal(reconstructed_tall, A_tall, tolerance = 1e-5)

  # Wide matrix (more columns than rows)
  A_wide <- matrix(rnorm(15), 3, 5)
  svd_wide <- svd(as_mlx(A_wide))
  U <- as.matrix(svd_wide$u)
  V <- as.matrix(svd_wide$v)
  reconstructed_wide <- U %*% diag(svd_wide$d) %*% t(V)
  expect_equal(reconstructed_wide, A_wide, tolerance = 1e-5)

  # Square matrix
  A_square <- matrix(rnorm(16), 4, 4)
  svd_square <- svd(as_mlx(A_square))
  U <- as.matrix(svd_square$u)
  V <- as.matrix(svd_square$v)
  reconstructed_square <- U %*% diag(svd_square$d) %*% t(V)
  expect_equal(reconstructed_square, A_square, tolerance = 1e-5)
})

test_that("svd.mlx U and V are orthogonal", {
  set.seed(707)
  A <- matrix(rnorm(20), 5, 4)

  svd_mlx <- svd(as_mlx(A))
  U <- as.matrix(svd_mlx$u)
  V <- as.matrix(svd_mlx$v)

  # U^T U should be identity
  expect_equal(t(U) %*% U, diag(4), tolerance = 1e-5)

  # V^T V should be identity
  expect_equal(t(V) %*% V, diag(4), tolerance = 1e-5)
})

test_that("svd.mlx preserves device and dtype", {
  old_device <- mlx_default_device()
  on.exit(mlx_default_device(old_device))

  mlx_default_device("gpu")

  set.seed(808)
  A <- matrix(rnorm(12), 3, 4)

  A_gpu <- as_mlx(A, device = "gpu", dtype = "float32")
  svd_gpu <- svd(A_gpu)

  expect_equal(svd_gpu$u$device, "gpu")
  expect_equal(svd_gpu$v$device, "gpu")
  expect_equal(svd_gpu$u$dtype, "float32")
  expect_equal(svd_gpu$v$dtype, "float32")

  # Check values still match
  svd_r <- svd(A)
  expect_equal(svd_gpu$d, svd_r$d, tolerance = 1e-4)
})

test_that("svd.mlx errors with invalid nu or nv", {
  set.seed(909)
  A <- matrix(rnorm(12), 3, 4)
  A_mlx <- as_mlx(A)

  # nu must be 0 or min(nrow, ncol)
  expect_error(svd(A_mlx, nu = 1), "nu = 0 or nu = min")
  expect_error(svd(A_mlx, nu = 2), "nu = 0 or nu = min")

  # nv must be 0 or min(nrow, ncol)
  expect_error(svd(A_mlx, nv = 1), "nv = 0 or nv = min")
  expect_error(svd(A_mlx, nv = 2), "nv = 0 or nv = min")
})

test_that("svd.mlx errors with LINPACK = TRUE", {
  set.seed(1010)
  A <- matrix(rnorm(12), 3, 4)

  expect_error(svd(as_mlx(A), LINPACK = TRUE), "LINPACK routines are not supported")
})

test_that("svd.mlx handles rank-deficient matrices", {
  # Create a rank-deficient matrix
  A <- matrix(c(1, 2, 2, 4, 3, 6), 3, 2)
  # This matrix has rank 1 (second column is 2x first column)

  svd_mlx <- svd(as_mlx(A))

  # Should have one large singular value and one near-zero
  expect_true(svd_mlx$d[1] > 1)
  expect_true(svd_mlx$d[2] < 1e-5)

  # Reconstruction should still work
  U <- as.matrix(svd_mlx$u)
  V <- as.matrix(svd_mlx$v)
  reconstructed <- U %*% diag(svd_mlx$d) %*% t(V)
  expect_equal(reconstructed, A, tolerance = 1e-5)
})

test_that("pinv matches analytical pseudoinverse for full-rank matrix", {
  set.seed(202)
  A <- matrix(rnorm(12), 4, 3)
  pseudo_r <- solve(t(A) %*% A) %*% t(A)

  pseudo_mlx <- pinv(as_mlx(A))
  expect_equal(as.matrix(pseudo_mlx), pseudo_r, tolerance = 1e-4)
})
