test_that("mlx_stack stacks tensors along arbitrary axes", {
  x <- as_mlx(matrix(1:4, 2, 2))
  y <- as_mlx(matrix(5:8, 2, 2))

  stacked <- mlx_stack(x, y, axis = 1)
  expect_equal(dim(stacked), c(2L, 2L, 2L))
  expected <- array(0, dim = c(2, dim(as.matrix(x))))
  expected[1, , ] <- as.matrix(x)
  expected[2, , ] <- as.matrix(y)
  expect_equal(as.array(stacked), expected, tolerance = 1e-6)

  stacked_last <- mlx_stack(list(x, y), axis = -1)
  expect_equal(dim(stacked_last), c(2L, 2L, 2L))
  arr <- array(0, dim = c(2, dim(as.matrix(x))))
  arr[1, , ] <- as.matrix(x)
  arr[2, , ] <- as.matrix(y)
  expected_last <- aperm(arr, c(2, 3, 1))
  expect_equal(as.array(stacked_last), expected_last, tolerance = 1e-6)
})

test_that("mlx_squeeze and mlx_expand_dims adjust shapes", {
  x <- as_mlx(array(1:4, dim = c(1, 2, 1, 2)))
  squeezed_all <- mlx_squeeze(x)
  expect_equal(dim(squeezed_all), c(2L, 2L))

  squeezed_axis <- mlx_squeeze(x, axis = c(1, 3))
  expect_equal(dim(squeezed_axis), c(2L, 2L))

  expanded <- mlx_expand_dims(squeezed_axis, axis = c(1, 3))
  expect_equal(dim(expanded), c(1L, 2L, 1L, 2L))
})

test_that("mlx_repeat repeats along axes", {
  x <- as_mlx(matrix(1:4, 2, 2))
  rep_cols <- mlx_repeat(x, repeats = 2, axis = 2)
  mat <- as.matrix(x)
  expected_cols <- matrix(0, nrow = nrow(mat), ncol = ncol(mat) * 2)
  for (j in seq_len(ncol(mat))) {
    expected_cols[, (2 * j - 1):(2 * j)] <- mat[, j]
  }
  expect_equal(as.matrix(rep_cols), expected_cols, tolerance = 1e-6)

  rep_flat <- mlx_repeat(x, repeats = 3)
  expect_equal(as.vector(as.matrix(rep_flat)), rep(as.vector(t(as.matrix(x))), each = 3))
})

test_that("mlx_tile tiles tensors", {
  x <- as_mlx(matrix(1:4, 2, 2))
  tiled <- mlx_tile(x, reps = c(2, 1))
  expected <- rbind(as.matrix(x), as.matrix(x))
  expect_equal(as.matrix(tiled), expected, tolerance = 1e-6)
})

test_that("mlx_roll shifts elements circularly", {
  x <- as_mlx(matrix(1:4, 2, 2))
  rolled_cols <- mlx_roll(x, shift = 1, axis = 2)
  expected_cols <- t(apply(as.matrix(x), 1, function(row) c(tail(row, 1), head(row, -1))))
  expect_equal(as.matrix(rolled_cols), expected_cols)

  rolled_flat <- mlx_roll(x, shift = -2)
  vec <- as.vector(t(as.matrix(x)))
  n <- length(vec)
  shift <- -2L %% n
  indices <- ((seq_len(n) - 1 - shift) %% n) + 1L
  expected_flat <- matrix(vec[indices], nrow = nrow(x), byrow = TRUE)
  expect_equal(as.matrix(rolled_flat), expected_flat)
})

test_that("mlx_moveaxis reorders axes", {
  arr <- array(seq_len(24), dim = c(2, 3, 4))
  x <- as_mlx(arr)

  moved_last <- mlx_moveaxis(x, source = 1, destination = 3)
  expect_equal(dim(moved_last), c(3L, 4L, 2L))
  expect_equal(as.array(moved_last), aperm(arr, c(2, 3, 1)), tolerance = 1e-6)

  moved_multi <- mlx_moveaxis(x, source = c(1, 3), destination = c(3, 1))
  expect_equal(as.array(moved_multi), aperm(arr, c(3, 2, 1)), tolerance = 1e-6)
})

test_that("aperm.mlx matches base behaviour", {
  arr <- array(runif(24), dim = c(2, 3, 4))
  x <- as_mlx(arr)

  perm <- c(2, 1, 3)
  permuted <- aperm(x, perm)
  expect_equal(as.array(permuted), aperm(arr, perm), tolerance = 1e-6)

  reversed <- aperm(x)
  expect_equal(as.array(reversed), aperm(arr), tolerance = 1e-6)
})

test_that("mlx_where acts like ifelse for tensors", {
  cond <- as_mlx(matrix(c(TRUE, FALSE, TRUE, FALSE), 2, 2))
  a <- as_mlx(matrix(1:4, 2, 2))
  b <- as_mlx(matrix(5:8, 2, 2))

  result <- mlx_where(cond, a, b)
  expect_equal(as.matrix(result), ifelse(as.matrix(cond), as.matrix(a), as.matrix(b)))

  # Broadcasting scalar
  result_scalar <- mlx_where(cond, 1, b)
  expect_equal(as.matrix(result_scalar), ifelse(as.matrix(cond), 1, as.matrix(b)))
})
