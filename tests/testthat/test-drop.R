test_that("drop.mlx removes singleton axes", {
  arr <- array(seq_len(6), dim = c(1, 3, 2, 1))
  x <- as_mlx(arr)

  dropped <- drop(x)

  expect_equal(mlx_dim(dropped), c(3L, 2L))
  expect_equal(as.array(dropped), drop(arr), tolerance = 1e-6)
})

test_that("drop.mlx is a no-op when no singleton axes exist", {
  mat <- matrix(1:6, nrow = 3)
  x <- as_mlx(mat)

  dropped <- drop(x)

  expect_equal(mlx_dim(dropped), c(3L, 2L))
  expect_equal(as.matrix(dropped), mat, tolerance = 1e-6)
})

test_that("drop.mlx preserves scalar payloads, dtype, and device", {
  scalar <- as_mlx(array(5, dim = c(1, 1, 1)))

  dropped <- drop(scalar)

  expect_equal(mlx_dim(dropped), integer(0))
  expect_equal(dropped$dtype, scalar$dtype)
  expect_equal(dropped$device, scalar$device)
  expect_equal(as.vector(dropped), 5, tolerance = 1e-6)
})
