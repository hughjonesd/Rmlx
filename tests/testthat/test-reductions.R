test_that("Summary group reductions work", {
  x <- as_mlx(matrix(1:4, 2, 2))
  y <- as_mlx(matrix(5:8, 2, 2))

  expect_equal(as.matrix(sum(x)), sum(1:4), tolerance = 1e-6)
  expect_equal(as.matrix(sum(x, y)), sum(c(1:4, 5:8)), tolerance = 1e-6)

  expect_equal(as.matrix(prod(x)), prod(1:4), tolerance = 1e-6)
  expect_equal(as.matrix(prod(x, 2)), prod(c(1:4, 2)), tolerance = 1e-6)

  bool_x <- x > 0
  expect_equal(as.vector(as.matrix(all(bool_x))), TRUE)
  expect_equal(as.vector(as.matrix(any(x > 10))), FALSE)
  expect_equal(as.vector(as.matrix(all(bool_x, x > 2))), FALSE)
})

test_that("mlx_sum and friends reduce axes", {
  x <- as_mlx(array(1:24, dim = c(2, 3, 4)))
  x_arr <- array(1:24, dim = c(2, 3, 4))

  expect_equal(as.matrix(mlx_sum(x, axis = 2)), apply(x_arr, c(1,3), sum))
  expect_equal(as.array(mlx_sum(x, axis = c(1, 3), drop = FALSE)),
               array(apply(x_arr, 2, sum), dim = c(1, 3, 1)))

  expect_equal(as.matrix(mlx_prod(x, axis = 3)), apply(x_arr, c(1,2), prod))

  bool_arr <- x > 12
  expect_equal(as.matrix(mlx_all(bool_arr, axis = 3)), apply(as.array(bool_arr), c(1,2), all))
  expect_equal(as.matrix(mlx_any(bool_arr, axis = 1)), apply(as.array(bool_arr), c(2,3), any))
})

test_that("mlx_mean/var/std support axes and ddof", {
  arr <- array(seq_len(18), dim = c(3, 3, 2))
  x <- as_mlx(arr)

  expect_equal(as.matrix(mlx_mean(x, axis = 3)), apply(arr, c(1,2), mean), tolerance = 1e-6)

  mean_keep <- mlx_mean(x, axis = 2, drop = FALSE)
  expect_equal(dim(as.array(mean_keep)), c(3, 1, 2))

  var0_expected <- apply(arr, c(1,2), function(v) mean((v - mean(v))^2))
  expect_equal(as.matrix(mlx_var(x, axis = 3)), var0_expected, tolerance = 1e-6)

  var1_expected <- apply(arr, c(1,2), stats::var)
  expect_equal(as.matrix(mlx_var(x, axis = 3, ddof = 1)), var1_expected, tolerance = 1e-6)

  std_expected <- sqrt(var0_expected)
  expect_equal(as.matrix(mlx_std(x, axis = 3)), std_expected, tolerance = 1e-6)

  overall_std <- as.vector(as.matrix(mlx_std(x, ddof = 1)))
  expect_equal(overall_std, stats::sd(as.vector(arr)), tolerance = 1e-6)
})
