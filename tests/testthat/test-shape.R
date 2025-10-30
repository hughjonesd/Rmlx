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
  expect_equal(as.vector(rep_flat), rep(as.vector(t(as.matrix(x))), each = 3))
})

test_that("mlx_tile tiles tensors", {
  x <- as_mlx(matrix(1:4, 2, 2))
  tiled <- mlx_tile(x, reps = c(2, 1))
  expected <- rbind(as.matrix(x), as.matrix(x))
  expect_equal(as.matrix(tiled), expected, tolerance = 1e-6)
})

test_that("mlx_pad pads tensors with various specifications", {
  x <- as_mlx(matrix(1:4, 2, 2))
  padded_scalar <- mlx_pad(x, pad_width = 1)
  expected_scalar <- matrix(0, nrow = 4, ncol = 4)
  expected_scalar[2:3, 2:3] <- as.matrix(x)
  expect_equal(as.matrix(padded_scalar), expected_scalar, tolerance = 1e-6)

  padded_axis <- mlx_pad(x, pad_width = c(0, 2), axes = 2, value = -1)
  expected_axis <- cbind(as.matrix(x), matrix(-1, nrow = 2, ncol = 2))
  expect_equal(as.matrix(padded_axis), expected_axis, tolerance = 1e-6)

  pad_list <- list(c(1, 0), c(0, 1))
  padded_list <- mlx_pad(x, pad_width = pad_list)
  expected_list <- rbind(
    c(0, 0, 0),
    c(1, 3, 0),
    c(2, 4, 0)
  )
  expect_equal(as.matrix(padded_list), expected_list, tolerance = 1e-6)
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

test_that("mlx_contiguous materializes a contiguous copy", {
  base <- matrix(1:12, nrow = 3, byrow = TRUE)
  x <- as_mlx(base)
  view <- mlx_swapaxes(x, axis1 = 1, axis2 = 2)

  contig <- mlx_contiguous(view)
  expect_equal(as.matrix(contig), as.matrix(view), tolerance = 1e-6)

  cpu_copy <- mlx_contiguous(view, device = "cpu")
  expect_equal(cpu_copy$device, "cpu")
  expect_equal(as.matrix(cpu_copy), as.matrix(view), tolerance = 1e-6)
})

row_major_vec <- function(arr) {
  dims <- dim(arr)
  if (is.null(dims)) {
    return(as.vector(arr))
  }
  res <- numeric(prod(dims))
  idx <- 1L
  current <- vector("list", length(dims))

  fill_vals <- function(axis) {
    if (axis > length(dims)) {
      res[idx] <<- do.call("[", c(list(arr), current))
      idx <<- idx + 1L
      return(invisible(NULL))
    }
    for (val in seq_len(dims[axis])) {
      current[[axis]] <<- val
      fill_vals(axis + 1L)
    }
  }

  fill_vals(1L)
  res
}

test_that("mlx_flatten collapses axes", {
  arr <- array(seq_len(12), dim = c(2, 3, 2))
  x <- as_mlx(arr)

  flat_all <- mlx_flatten(x)
  expect_equal(mlx_dim(flat_all), c(12L))
  expect_equal(as.vector(flat_all), row_major_vec(arr), tolerance = 1e-6)

  flat_middle <- mlx_flatten(x, start_axis = 2, end_axis = 3)
  expect_equal(mlx_dim(flat_middle), c(2L, 6L))
  expected <- matrix(row_major_vec(arr), nrow = dim(arr)[1], byrow = TRUE)
  expect_equal(as.matrix(flat_middle), expected, tolerance = 1e-6)

  expect_error(mlx_flatten(x, start_axis = 3, end_axis = 2))
})

test_that("mlx_swapaxes swaps specified axes", {
  arr <- array(seq_len(24), dim = c(2, 3, 4))
  x <- as_mlx(arr)

  swapped <- mlx_swapaxes(x, axis1 = 1, axis2 = 3)
  expect_equal(mlx_dim(swapped), c(4L, 3L, 2L))
  expect_equal(as.array(swapped), aperm(arr, c(3, 2, 1)), tolerance = 1e-6)

  swapped_neg <- mlx_swapaxes(x, axis1 = -2, axis2 = -1)
  expect_equal(as.array(swapped_neg), aperm(arr, c(1, 3, 2)), tolerance = 1e-6)
})

test_that("mlx_meshgrid creates coordinate tensors", {
  x <- as_mlx(0:1)
  y <- as_mlx(0:2)

  grids_xy <- mlx_meshgrid(x, y, indexing = "xy")
  expect_length(grids_xy, 2L)
  expect_equal(mlx_dim(grids_xy[[1]]), c(3L, 2L))
  expect_equal(mlx_dim(grids_xy[[2]]), c(3L, 2L))
  expect_equal(as.matrix(grids_xy[[1]]), outer(0:2, 0:1, function(y, x) x), tolerance = 1e-6)
  expect_equal(as.matrix(grids_xy[[2]]), outer(0:2, 0:1, function(y, x) y), tolerance = 1e-6)

  grids_ij <- mlx_meshgrid(x, y, indexing = "ij")
  expect_equal(mlx_dim(grids_ij[[1]]), c(2L, 3L))
  expect_equal(mlx_dim(grids_ij[[2]]), c(2L, 3L))
  expect_equal(as.matrix(grids_ij[[1]]), outer(0:1, 0:2, function(x, y) x), tolerance = 1e-6)
  expect_equal(as.matrix(grids_ij[[2]]), outer(0:1, 0:2, function(x, y) y), tolerance = 1e-6)

  sparse <- mlx_meshgrid(x, y, sparse = TRUE, indexing = "xy")
  expect_equal(mlx_dim(sparse[[1]]), c(1L, 2L))
  expect_equal(mlx_dim(sparse[[2]]), c(3L, 1L))
})

test_that("mlx_broadcast_to expands singleton dimensions", {
  x <- as_mlx(matrix(1:3, nrow = 1))
  broadcasted <- mlx_broadcast_to(x, c(4, 3))

  expect_equal(mlx_dim(broadcasted), c(4L, 3L))
  expected <- matrix(rep(1:3, times = 4), nrow = 4, byrow = TRUE)
  expect_equal(as.matrix(broadcasted), expected, tolerance = 1e-6)
})

test_that("mlx_broadcast_arrays aligns shapes", {
  a <- as_mlx(matrix(1:3, nrow = 1))
  b <- as_mlx(matrix(c(10, 20, 30), ncol = 1))

  outs <- mlx_broadcast_arrays(a, b)
  expect_length(outs, 2L)
  expect_equal(mlx_dim(outs[[1]]), c(3L, 3L))
  expect_equal(mlx_dim(outs[[2]]), c(3L, 3L))

  expected_a <- matrix(rep(1:3, times = 3), nrow = 3, byrow = TRUE)
  expected_b <- matrix(c(10, 20, 30), nrow = 3, ncol = 3)
  expect_equal(as.matrix(outs[[1]]), expected_a, tolerance = 1e-6)
  expect_equal(as.matrix(outs[[2]]), expected_b, tolerance = 1e-6)
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

test_that("mlx_split divides tensors into equal parts", {
  arr <- matrix(1:6, nrow = 3, byrow = TRUE)
  x <- as_mlx(arr)

  parts <- mlx_split(x, sections = 3, axis = 1)
  expect_equal(length(parts), 3L)
  expect_true(all(vapply(parts, inherits, logical(1), what = "mlx")))
  expect_equal(lapply(parts, dim), list(c(1L, 2L), c(1L, 2L), c(1L, 2L)))
  expect_equal(as.matrix(parts[[2]]), arr[2, , drop = FALSE], tolerance = 1e-6)
})

test_that("mlx_split supports custom split points", {
  arr <- array(1:12, dim = c(3, 4))
  x <- as_mlx(arr)

  parts <- mlx_split(x, sections = c(1, 3), axis = 2)
  expect_equal(length(parts), 3L)
  expect_equal(dim(parts[[1]]), c(3L, 1L))
  expect_equal(dim(parts[[2]]), c(3L, 2L))
  expect_equal(dim(parts[[3]]), c(3L, 1L))

  reconstructed <- do.call(cbind, lapply(parts, as.matrix))
  expect_equal(reconstructed, as.matrix(x), tolerance = 1e-6)
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
