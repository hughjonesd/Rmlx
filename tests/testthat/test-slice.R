test_that("slicing with strides preserves dimensionality", {
  x <- matrix(1:9, 3, 3)
  x_mlx <- as_mlx(x)

  sliced <- x_mlx[seq(1, 3, by = 2), ]
  expect_equal(dim(sliced), c(2L, 3L))

  sliced_mat <- as.matrix(sliced)
  expect_equal(sliced_mat, x[seq(1, 3, by = 2), , drop = FALSE], tolerance = 1e-6)
})
