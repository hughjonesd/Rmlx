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

  expect_equal(as.matrix(mlx_sum(x, axis = 2)), apply(array(1:24, c(2,3,4)), c(1,3), sum))
  expect_equal(as.matrix(mlx_sum(x, axis = c(1, 3), drop = FALSE)),
               array(apply(array(1:24, c(2,3,4)), 2, sum), dim = c(1, 3, 1)))

  expect_equal(as.matrix(mlx_prod(x, axis = 3)), apply(array(1:24, c(2,3,4)), c(1,2), prod))

  bool_arr <- x > 12
  expect_equal(as.matrix(mlx_all(bool_arr, axis = 3)), apply(as.array(bool_arr), c(1,2), all))
  expect_equal(as.matrix(mlx_any(bool_arr, axis = 1)), apply(as.array(bool_arr), c(2,3), any))
})
