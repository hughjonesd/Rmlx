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

  expect_equal(as.numeric(x_mlx), x_r, tolerance = 1e-5)
})

test_that("solve works when A is_mlx and b is R matrix", {
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
  expect_equal(mlx_dtype(x_gpu), "float32")
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
  expect_equal(mlx_dtype(R_gpu), "float32")
  expect_equal(as.matrix(R_gpu), chol(spd), tolerance = 1e-4)
})

test_that("chol.mlx errors with pivot = TRUE", {
  set.seed(567)
  A <- matrix(rnorm(9), 3, 3)
  spd <- crossprod(A) + diag(3) * 0.1

  expect_error(chol(as_mlx(spd), pivot = TRUE), "pivoted Cholesky is not supported")
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
  d <- as.vector(svd_mlx$d)
  V <- as.matrix(svd_mlx$v)

  # the subsets are because mlx does "full" svd
  reconstructed <- U %*% diag(d) %*% t(V[, 1:3])
  expect_equal(reconstructed, A, tolerance = 1e-4)
})

test_that("svd.mlx with nu=0 and nv=0 returns only singular values", {
  set.seed(404)
  A <- matrix(rnorm(12), 3, 4)
  svd_mlx <- svd(as_mlx(A), nu = 0, nv = 0)

  expect_null(svd_mlx$u)
  expect_true(is_mlx(svd_mlx$d))
  expect_null(svd_mlx$v)
  expect_equal(length(svd_mlx$d), 3)
})

test_that("svd.mlx singular values match base R", {
  set.seed(505)
  A <- matrix(rnorm(20), 4, 5)

  svd_r <- svd(A)
  svd_mlx <- svd(as_mlx(A))

  expect_equal(as.vector(svd_mlx$d), svd_r$d, tolerance = 1e-5)
})

test_that("mlx_kron matches base kronecker", {
  set.seed(135)
  A <- matrix(rnorm(4), 2, 2)
  B <- matrix(rnorm(6), 2, 3)

  kron_mlx <- mlx_kron(A, B)
  kron_r <- kronecker(A, B)

  expect_equal(as.matrix(kron_mlx), kron_r, tolerance = 1e-6)
})

test_that("kronecker methods dispatch for mlx", {
  A <- as_mlx(matrix(1:4, 2, 2))
  B <- as_mlx(matrix(5:8, 2, 2))

  res <- kronecker(A, B)
  expect_s3_class(res, "mlx")
  expect_equal(as.matrix(res), kronecker(as.matrix(A), as.matrix(B)), tolerance = 1e-6)

  res_mixed1 <- kronecker(A, matrix(1:4, 2, 2))
  res_mixed2 <- kronecker(matrix(1:4, 2, 2), B)

  expect_equal(as.matrix(res_mixed1), kronecker(as.matrix(A), matrix(1:4, 2, 2)), tolerance = 1e-6)
  expect_equal(as.matrix(res_mixed2), kronecker(matrix(1:4, 2, 2), as.matrix(B)), tolerance = 1e-6)
})

test_that("svd.mlx works with different matrix dimensions", {
  set.seed(606)

  # Tall matrix (more rows than columns)
  A_tall <- matrix(rnorm(15), 5, 3)
  svd_tall <- svd(as_mlx(A_tall))
  U <- as.matrix(svd_tall$u)
  V <- as.matrix(svd_tall$v)
  d <- as.vector(svd_tall$d)
  # full svd strikes again:
  reconstructed_tall <- U[, 1:3] %*% diag(d) %*% t(V)
  expect_equal(reconstructed_tall, A_tall, tolerance = 1e-5)

  # Wide matrix (more columns than rows)
  A_wide <- matrix(rnorm(15), 3, 5)
  svd_wide <- svd(as_mlx(A_wide))
  U <- as.matrix(svd_wide$u)
  V <- as.matrix(svd_wide$v)
  d <- as.vector(svd_wide$d)

  reconstructed_wide <- U %*% diag(d) %*% t(V[, 1:3])
  expect_equal(reconstructed_wide, A_wide, tolerance = 1e-5)

  # Square matrix
  A_square <- matrix(rnorm(16), 4, 4)
  svd_square <- svd(as_mlx(A_square))
  U <- as.matrix(svd_square$u)
  V <- as.matrix(svd_square$v)
  d <- as.vector(svd_square$d)

  reconstructed_square <- U %*% diag(d) %*% t(V)
  expect_equal(reconstructed_square, A_square, tolerance = 1e-5)
})

test_that("svd.mlx U and V are orthogonal", {
  set.seed(707)
  A <- matrix(rnorm(20), 5, 4)

  svd_mlx <- svd(as_mlx(A))
  U <- as.matrix(svd_mlx$u)
  V <- as.matrix(svd_mlx$v)

  # U^T U should be identity (for the first 4 columns)
  expect_equal(t(U[, 1:4]) %*% U[, 1:4], diag(4), tolerance = 1e-5)

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
  expect_equal(mlx_dtype(svd_gpu$u), "float32")
  expect_equal(mlx_dtype(svd_gpu$v), "float32")

  # Check values still match
  svd_r <- svd(A)
  expect_equal(as.vector(svd_gpu$d), svd_r$d, tolerance = 1e-4)
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

test_that("svd.mlx handles rank-deficient matrices", {
  # Create a rank-deficient matrix
  A <- matrix(c(1, 2, 2, 4, 3, 6), 3, 2, byrow = TRUE)
  # This matrix has rank 1 (second column is 2x first column)

  svd_mlx <- svd(as_mlx(A))

  # Should have one large singular value and one near-zero
  svd_mlx_d <- as.vector(svd_mlx$d)
  expect_true(svd_mlx_d[1] > 1)
  expect_true(svd_mlx_d[2] < 1e-5)

  # Reconstruction should still work
  U <- as.matrix(svd_mlx$u)
  V <- as.matrix(svd_mlx$v)
  # the subsets are because mlx does "full" svd
  reconstructed <- U[, 1:2] %*% diag(svd_mlx_d) %*% t(V)
  expect_equal(reconstructed, A, tolerance = 1e-5)
})

test_that("pinv matches analytical pseudoinverse for full-rank matrix", {
  set.seed(202)
  A <- matrix(rnorm(12), 4, 3)
  pseudo_r <- solve(t(A) %*% A) %*% t(A)

  pseudo_mlx <- pinv(as_mlx(A))
  expect_equal(as.matrix(pseudo_mlx), pseudo_r, tolerance = 1e-4)
})

test_that("mlx_inv computes matrix inverse", {
  set.seed(301)
  A <- matrix(rnorm(16), 4, 4)

  A_inv_mlx <- mlx_inv(as_mlx(A))
  A_inv_r <- solve(A)

  expect_equal(as.matrix(A_inv_mlx), A_inv_r, tolerance = 1e-5)

  # Verify: A %*% A_inv should be identity
  I_mlx <- as_mlx(A) %*% A_inv_mlx
  I_expected <- diag(4)
  expect_equal(as.matrix(I_mlx), I_expected, tolerance = 1e-5)
})

test_that("mlx_inv works with different matrix sizes", {
  set.seed(302)

  # 2x2 matrix
  A2 <- matrix(rnorm(4), 2, 2)
  A2_inv_mlx <- mlx_inv(as_mlx(A2))
  expect_equal(as.matrix(A2_inv_mlx), solve(A2), tolerance = 1e-5)

  # 5x5 matrix
  A5 <- matrix(rnorm(25), 5, 5)
  A5_inv_mlx <- mlx_inv(as_mlx(A5))
  expect_equal(as.matrix(A5_inv_mlx), solve(A5), tolerance = 1e-5)
})

test_that("mlx_tri_inv computes triangular matrix inverse", {
  set.seed(303)
  A <- matrix(rnorm(16), 4, 4)

  # Lower triangular
  L <- A
  L[upper.tri(L)] <- 0
  L_inv_mlx <- mlx_tri_inv(as_mlx(L), upper = FALSE)
  L_inv_r <- solve(L)
  expect_equal(as.matrix(L_inv_mlx), L_inv_r, tolerance = 1e-5)

  # Upper triangular
  U <- A
  U[lower.tri(U)] <- 0
  U_inv_mlx <- mlx_tri_inv(as_mlx(U), upper = TRUE)
  U_inv_r <- solve(U)
  expect_equal(as.matrix(U_inv_mlx), U_inv_r, tolerance = 1e-5)
})

test_that("mlx_cholesky_inv computes inverse via Cholesky", {
  set.seed(304)
  # Create a positive definite matrix
  A <- matrix(rnorm(16), 4, 4)
  A <- t(A) %*% A  # Make it positive definite

  # R's chol() returns upper triangular by default
  U <- chol(A)

  # Get inverse from Cholesky factor
  A_inv_mlx <- mlx_cholesky_inv(as_mlx(U), upper = TRUE)
  A_inv_r <- solve(A)

  expect_equal(as.matrix(A_inv_mlx), A_inv_r, tolerance = 1e-4)

  # Verify inverse
  I_mlx <- as_mlx(A) %*% A_inv_mlx
  expect_equal(as.matrix(I_mlx), diag(4), tolerance = 1e-3)
})

test_that("mlx_cholesky_inv works with lower triangle", {
  set.seed(305)
  A <- matrix(rnorm(9), 3, 3)
  A <- t(A) %*% A  # Positive definite

  # Get upper triangular Cholesky factor and transpose to get lower
  U <- chol(A)
  L <- t(U)

  A_inv_mlx <- mlx_cholesky_inv(as_mlx(L), upper = FALSE)
  A_inv_r <- solve(A)

  expect_equal(as.matrix(A_inv_mlx), A_inv_r, tolerance = 1e-4)
})

test_that("chol2inv.mlx works like base R chol2inv", {
  set.seed(305)
  A <- matrix(rnorm(16), 4, 4)
  A <- t(A) %*% A  # Make it positive definite

  # Base R
  U_r <- chol(A)
  A_inv_r <- chol2inv(U_r)

  # MLX
  A_mlx <- as_mlx(A)
  U_mlx <- chol(A_mlx)
  A_inv_mlx <- chol2inv(U_mlx)

  expect_equal(as.matrix(A_inv_mlx), A_inv_r, tolerance = 1e-4)

  # Verify inverse
  I_mlx <- A_mlx %*% A_inv_mlx
  expect_equal(as.matrix(I_mlx), diag(4), tolerance = 1e-3)
})

test_that("mlx_lu returns P, L and U factors", {
  set.seed(306)
  A <- matrix(rnorm(16), 4, 4)

  lu_mlx <- mlx_lu(as_mlx(A))

  expect_true(is.list(lu_mlx))
  expect_true("p" %in% names(lu_mlx))
  expect_true("l" %in% names(lu_mlx))
  expect_true("u" %in% names(lu_mlx))

  P <- as.vector(lu_mlx$p)
  L <- as.matrix(lu_mlx$l)
  U <- as.matrix(lu_mlx$u)

  # L should be lower triangular with 1s on diagonal
  expect_true(all(L[upper.tri(L)] == 0))
  expect_equal(diag(L), rep(1, 4), tolerance = 1e-10)

  # U should be upper triangular
  expect_true(all(U[lower.tri(U)] == 0))

  # P should be pivot indices (length matching dimensions)
  expect_equal(length(P), 4)
})

test_that("mlx_lu works with rectangular matrices", {
  set.seed(307)

  # Tall matrix
  A_tall <- matrix(rnorm(12), 4, 3)
  lu_tall <- mlx_lu(as_mlx(A_tall))
  # Check that we get results with expected structure
  expect_true(is.list(lu_tall))
  expect_true(all(c("p", "l", "u") %in% names(lu_tall)))

  # Wide matrix
  A_wide <- matrix(rnorm(12), 3, 4)
  lu_wide <- mlx_lu(as_mlx(A_wide))
  expect_true(is.list(lu_wide))
  expect_true(all(c("p", "l", "u") %in% names(lu_wide)))
})
