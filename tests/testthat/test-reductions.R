test_that("Summary group reductions work", {
  x <- mlx_matrix(1:4, 2, 2)
  y <- mlx_matrix(5:8, 2, 2)

  expect_equal(as.vector(sum(x)), sum(1:4), tolerance = 1e-6)
  expect_equal(as.vector(sum(x, y)), sum(c(1:4, 5:8)), tolerance = 1e-6)

  expect_equal(as.vector(prod(x)), prod(1:4), tolerance = 1e-6)
  expect_equal(as.vector(prod(x, 2)), prod(c(1:4, 2)), tolerance = 1e-6)

  bool_x <- x > 0
  expect_equal(all(bool_x), TRUE)
  expect_equal(any(x > 10), FALSE)
  expect_equal(all(bool_x, x > 2), FALSE)
})

test_that("mlx_sum and friends reduce axes", {
  x <- mlx_array(1:24, dim = c(2, 3, 4))
  x_arr <- array(1:24, dim = c(2, 3, 4))

  expect_equal(as.matrix(mlx_sum(x, axes = 2)), apply(x_arr, c(1,3), sum))
  expect_equal(as.array(mlx_sum(x, axes = c(1, 3), drop = FALSE)),
               array(apply(x_arr, 2, sum), dim = c(1, 3, 1)))

  expect_equal(as.matrix(mlx_prod(x, axes = 3)), apply(x_arr, c(1,2), prod))

  bool_arr <- x > 12
  expect_equal(as.matrix(mlx_all(bool_arr, axes = 3)), apply(as.array(bool_arr), c(1,2), all))
  expect_equal(as.matrix(mlx_any(bool_arr, axes = 1)), apply(as.array(bool_arr), c(2,3), any))
})

test_that("mlx_mean/var/std support axes and ddof", {
  arr <- array(seq_len(18), dim = c(3, 3, 2))
  x <- as_mlx(arr)

  expect_equal(as.matrix(mlx_mean(x, axes = 3)), apply(arr, c(1,2), mean), tolerance = 1e-6)

  mean_keep <- mlx_mean(x, axes = 2, drop = FALSE)
  expect_equal(dim(as.array(mean_keep)), c(3, 1, 2))

  var0_expected <- apply(arr, c(1,2), function(v) mean((v - mean(v))^2))
  expect_equal(as.matrix(mlx_var(x, axes = 3)), var0_expected, tolerance = 1e-6)

  var1_expected <- apply(arr, c(1,2), stats::var)
  expect_equal(as.matrix(mlx_var(x, axes = 3, ddof = 1)), var1_expected, tolerance = 1e-6)

  std_expected <- sqrt(var0_expected)
  expect_equal(as.matrix(mlx_std(x, axes = 3)), std_expected, tolerance = 1e-6)

  overall_std <- as.vector(mlx_std(x, ddof = 1))
  expect_equal(overall_std, stats::sd(as.vector(arr)), tolerance = 1e-6)
})

test_that("mlx_cumsum computes cumulative sum", {
  # 1D case
  x <- as_mlx(1:5)
  result <- mlx_cumsum(x)
  expect_equal(as.vector(result), cumsum(1:5))

  # 2D case with axis
  mat <- matrix(1:12, 3, 4)
  x <- as_mlx(mat)

  # Cumsum along rows (axis 1)
  result_rows <- mlx_cumsum(x, axis = 1)
  expect_equal(as.matrix(result_rows), apply(mat, 2, cumsum))

  # Cumsum along columns (axis 2)
  result_cols <- mlx_cumsum(x, axis = 2)
  expect_equal(as.matrix(result_cols), t(apply(mat, 1, cumsum)))

  # Reverse cumsum
  result_rev <- mlx_cumsum(x, axis = 1, reverse = TRUE)
  expected_rev <- apply(mat, 2, function(col) rev(cumsum(rev(col))))
  expect_equal(as.matrix(result_rev), expected_rev)

  # Exclusive cumsum (not inclusive)
  result_excl <- mlx_cumsum(x, axis = 1, inclusive = FALSE)
  expected_excl <- apply(mat, 2, function(col) c(0, cumsum(col[-length(col)])))
  expect_equal(as.matrix(result_excl), expected_excl)
})

test_that("mlx_cumprod computes cumulative product", {
  # 1D case
  x <- as_mlx(1:5)
  result <- mlx_cumprod(x)
  expect_equal(as.vector(result), cumprod(1:5))

  # 2D case with axis
  mat <- matrix(1:12, 3, 4)
  x <- as_mlx(mat)

  # Cumprod along rows (axis 1)
  result_rows <- mlx_cumprod(x, axis = 1)
  expect_equal(as.matrix(result_rows), apply(mat, 2, cumprod))

  # Cumprod along columns (axis 2)
  result_cols <- mlx_cumprod(x, axis = 2)
  expect_equal(as.matrix(result_cols), t(apply(mat, 1, cumprod)))

  # Reverse cumprod
  result_rev <- mlx_cumprod(x, axis = 1, reverse = TRUE)
  expected_rev <- apply(mat, 2, function(col) rev(cumprod(rev(col))))
  expect_equal(as.matrix(result_rev), expected_rev)

  # Exclusive cumprod (not inclusive)
  result_excl <- mlx_cumprod(x, axis = 1, inclusive = FALSE)
  expected_excl <- apply(mat, 2, function(col) c(1, cumprod(col[-length(col)])))
  expect_equal(as.matrix(result_excl), expected_excl)
})

test_that("mlx_quantile computes sample quantiles", {
  # Simple vector case
  x <- as_mlx(1:10)
  result <- mlx_quantile(x, 0.5)
  expected <- quantile(1:10, 0.5, type = 7)
  expect_equal(as.numeric(result), as.numeric(expected), tolerance = 1e-6)

  # Multiple probabilities
  result_multi <- mlx_quantile(x, c(0.25, 0.5, 0.75))
  expected_multi <- quantile(1:10, c(0.25, 0.5, 0.75), type = 7)
  expect_equal(as.numeric(result_multi), as.numeric(expected_multi), tolerance = 1e-6)

  # Quartiles
  result_quartiles <- quantile(x, probs = c(0, 0.25, 0.5, 0.75, 1))
  expected_quartiles <- quantile(1:10, probs = c(0, 0.25, 0.5, 0.75, 1), type = 7)
  expect_equal(as.numeric(result_quartiles), as.numeric(expected_quartiles), tolerance = 1e-6)

  # Edge cases
  expect_equal(as.numeric(mlx_quantile(x, 0)), 1, tolerance = 1e-6)
  expect_equal(as.numeric(mlx_quantile(x, 1)), 10, tolerance = 1e-6)

  # Random data
  set.seed(123)
  vec <- rnorm(100)
  x_mlx <- as_mlx(vec)
  result_random <- mlx_quantile(x_mlx, c(0.1, 0.5, 0.9))
  expected_random <- quantile(vec, c(0.1, 0.5, 0.9), type = 7)
  expect_equal(as.numeric(result_random), as.numeric(expected_random), tolerance = 1e-6)
})

test_that("mlx_quantile handles edge cases", {
  # Single element
  x_single <- as_mlx(5)
  expect_equal(as.numeric(mlx_quantile(x_single, 0.5)), 5, tolerance = 1e-6)
  expect_equal(as.numeric(mlx_quantile(x_single, c(0, 0.5, 1))), c(5, 5, 5), tolerance = 1e-6)

  # Two elements
  x_two <- as_mlx(c(1, 2))
  expect_equal(as.numeric(mlx_quantile(x_two, 0.5)), 1.5, tolerance = 1e-6)
  expect_equal(as.numeric(mlx_quantile(x_two, c(0, 1))), c(1, 2), tolerance = 1e-6)
})

test_that("mlx_quantile works with axis parameter", {
  # Matrix case: compute quantiles along axis 1 (columns)
  mat <- matrix(1:9, 3, 3, byrow = TRUE)
  x <- as_mlx(mat)
  result <- mlx_quantile(x, probs = c(1/3, 2/3), axis = 1)

  # Expected: quantiles for each column
  expected <- matrix(c(3, 5, 4, 6, 5, 7), nrow = 2, ncol = 3)
  expect_equal(as.matrix(result), expected, tolerance = 1e-6)

  # Single quantile with axis
  result_single <- mlx_quantile(x, probs = 0.5, axis = 1)
  expected_single <- apply(mat, 2, quantile, probs = 0.5, type = 7)
  expect_equal(as.numeric(result_single), expected_single, tolerance = 1e-6)

  # Compare with column-wise quantiles
  for (j in 1:3) {
    col_data <- mat[, j]
    for (p in c(0.25, 0.5, 0.75)) {
      result_p <- mlx_quantile(x, probs = p, axis = 1)
      expected_p <- unname(quantile(col_data, probs = p, type = 7))
      expect_equal(as.numeric(result_p)[j], expected_p, tolerance = 1e-6)
    }
  }

  # Test drop parameter
  result_no_drop <- mlx_quantile(x, probs = 0.5, axis = 1, drop = FALSE)
  expect_equal(mlx_shape(result_no_drop), c(1L, 3L))

  result_drop <- mlx_quantile(x, probs = 0.5, axis = 1, drop = TRUE)
  expect_null(dim(result_drop))
  expect_equal(mlx_shape(result_drop), 3L)
  expect_equal(as.vector(result_drop), apply(mat, 2, median), tolerance = 1e-6)

  # drop should not affect multiple quantiles
  result_multi <- mlx_quantile(x, probs = c(0.25, 0.75), axis = 1, drop = TRUE)
  expect_equal(mlx_shape(result_multi), c(2L, 3L))
})
