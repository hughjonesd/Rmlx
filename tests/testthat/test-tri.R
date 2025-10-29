test_that("mlx_tri matches lower triangular mask", {
  tri_res <- mlx_tri(3, 4, k = 0)
  expect_s3_class(tri_res, "mlx")

  expected <- matrix(0, 3, 4)
  mask <- col(expected) - row(expected) <= 0
  expected[mask] <- 1

  expect_equal(as.matrix(tri_res), expected)
})

test_that("mlx_tri supports offsets and rectangular shapes", {
  tri_res <- mlx_tri(4, 3, k = -1)
  expected <- matrix(0, 4, 3)
  mask <- col(expected) - row(expected) <= -1
  expected[mask] <- 1
  expect_equal(as.matrix(tri_res), expected)

  tri_upper <- mlx_tri(2, 5, k = 2)
  expected_upper <- matrix(0, 2, 5)
  mask_upper <- col(expected_upper) - row(expected_upper) <= 2
  expected_upper[mask_upper] <- 1
  expect_equal(as.matrix(tri_upper), expected_upper)
})

test_that("mlx_tril zeroes upper triangle like base lower.tri", {
  x <- matrix(1:9, 3, 3)
  x_mlx <- as_mlx(x)

  res <- mlx_tril(x_mlx)
  expect_s3_class(res, "mlx")

  expected <- x
  expected[col(expected) - row(expected) > 0] <- 0
  expect_equal(as.matrix(res), expected)

  res_k1 <- mlx_tril(x_mlx, k = 1)
  expected_k1 <- x
  expected_k1[col(expected_k1) - row(expected_k1) > 1] <- 0
  expect_equal(as.matrix(res_k1), expected_k1)
})

test_that("mlx_triu zeroes lower triangle like base upper.tri", {
  x <- matrix(1:9, 3, 3)
  x_mlx <- as_mlx(x)

  res <- mlx_triu(x_mlx)
  expect_s3_class(res, "mlx")

  expected <- x
  expected[row(expected) - col(expected) > 0] <- 0
  expect_equal(as.matrix(res), expected)

  res_km1 <- mlx_triu(x_mlx, k = -1)
  expected_km1 <- x
  expected_km1[col(expected_km1) - row(expected_km1) < -1] <- 0
  expect_equal(as.matrix(res_km1), expected_km1)
})
