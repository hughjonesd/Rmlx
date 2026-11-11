test_that("batched matmul broadcasts leading dimensions", {
  x_r <- array(seq_len(24) / 10, c(2, 3, 4))
  y_r <- matrix(seq_len(20) / 10, 4, 5)
  actual <- as.array(mlx_matmul(as_mlx(x_r), as_mlx(y_r)))
  expected <- array(0, c(2, 3, 5))
  for (i in 1:2) expected[i, , ] <- x_r[i, , ] %*% y_r
  expect_equal(actual, expected, tolerance = 1e-6)
})

test_that("batched matmul supports vector and matrix inputs", {
  expect_equal(as.numeric(mlx_matmul(as_mlx(1:3), as_mlx(4:6))), 32)
  x <- as_mlx(matrix(1:6, 2, 3))
  expect_equal(as.numeric(mlx_matmul(x, as_mlx(1:3))), as.numeric(matrix(1:6, 2, 3) %*% 1:3))
  expect_error(x %*% as_mlx(1:3), "2D matrices")
  expect_error(mlx_matmul(mlx_scalar(1), x), "at least one dimension")
})

test_that("batched matmul gradients accumulate broadcast batches", {
  x <- as_mlx(array(1, c(2, 3, 4)), dtype = "float32")
  w <- as_mlx(matrix(1, 4, 5), dtype = "float32")
  loss <- function(w, x) sum(mlx_matmul(x, w))
  gradient <- mlx_grad(loss, w, x, argnums = 1L)[[1]]
  expect_equal(as.matrix(gradient), matrix(6, 4, 5), tolerance = 1e-6)
})
