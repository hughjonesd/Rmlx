test_that("single logical index flattens like base R", {
  base_mat <- matrix((-4):3, nrow = 2)
  mask <- base_mat < 0
  mat <- base_mat
  expected <- mat
  expected[mask] <- 0
  x <- as_mlx(mat)

  x[mask] <- 0
  expect_equal(as.matrix(x), expected)

  mat <- base_mat
  mask_mlx <- as_mlx(mask)
  x_mlx_mask <- as_mlx(mat)
  x_mlx_mask[mask_mlx] <- 0
  expect_equal(as.matrix(x_mlx_mask), expected)
})

test_that("subset assignment with numeric indices matches base R", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)

  x[1, 2] <- 42
  mat[1, 2] <- 42
  expect_equal(as.matrix(x), mat, tolerance = 1e-6)

  x[2, ] <- c(10, 11, 12)
  mat[2, ] <- c(10, 11, 12)
  expect_equal(as.matrix(x), mat, tolerance = 1e-6)

  x[, 3] <- 100
  mat[, 3] <- 100
  expect_equal(as.matrix(x), mat, tolerance = 1e-6)
})

test_that("subset assignment accepts mlx replacement arrays", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)
  repl <- matrix(seq_len(4), nrow = 2)
  repl_mlx <- as_mlx(repl)

  x[1:2, 1:2] <- repl_mlx
  mat[1:2, 1:2] <- repl

  expect_equal(as.matrix(x), mat, tolerance = 1e-6)

  repl_alt <- as_mlx(repl, dtype = "float32", device = x$device)
  x[1:2, 1:2] <- repl_alt
  mat[1:2, 1:2] <- repl

  expect_equal(as.matrix(x), mat, tolerance = 1e-6)
})

test_that("vector subset assignment updates the correct element", {
  base_vec <- 1:5
  mlx_vec <- mlx_vector(base_vec)

  mlx_vec[1] <- 2
  base_vec[1] <- 2

  expect_equal(as.vector(mlx_vec), base_vec, tolerance = 1e-6)
})

test_that("vector subset assignment handles mlx and logical indices", {
  base_vec <- 1:6
  mlx_vec <- mlx_vector(base_vec)

  idx_mlx <- as_mlx(c(2L, 5L))
  mlx_vec[idx_mlx] <- c(20, 50)
  base_vec[c(2, 5)] <- c(20, 50)
  expect_equal(as.vector(mlx_vec), base_vec, tolerance = 1e-6)

  logical_mask <- c(FALSE, TRUE, FALSE, TRUE, FALSE, TRUE)
  logical_mask_mlx <- as_mlx(logical_mask)
  mlx_vec[logical_mask_mlx] <- 99
  base_vec[logical_mask] <- 99
  expect_equal(as.vector(mlx_vec), base_vec, tolerance = 1e-6)

  neg_idx <- as_mlx(-6L)
  mlx_vec[neg_idx] <- 42
  base_vec[-6] <- 42

  expect_equal(as.vector(mlx_vec), base_vec, tolerance = 1e-6)
})

test_that("subset assignment with logical masks behaves like base R", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)

  row_mask <- c(TRUE, FALSE, TRUE)
  col_mask <- c(FALSE, TRUE, TRUE)

  x[row_mask, col_mask] <- c(5, 6, 7, 8)
  mat[row_mask, col_mask] <- c(5, 6, 7, 8)

  expect_equal(as.matrix(x), mat, tolerance = 1e-6)

  x_mlx_mask <- as_mlx(mat)
  row_mask_mlx <- as_mlx(row_mask)
  col_mask_mlx <- as_mlx(col_mask)
  x_mlx_mask[row_mask_mlx, col_mask_mlx] <- as_mlx(c(9, 8, 7, 6))
  mat[row_mask, col_mask] <- c(9, 8, 7, 6)
  expect_equal(as.matrix(x_mlx_mask), mat, tolerance = 1e-6)
})

test_that("non-contiguous numeric assignment works without fast path", {
  old <- getOption("Rmlx_use_slice_fast_path")
  on.exit(options(Rmlx_use_slice_fast_path = old), add = TRUE)
  options(Rmlx_use_slice_fast_path = FALSE)

  mat <- matrix(seq_len(12 * 15), 12, 15)
  x <- as_mlx(mat)

  set.seed(20251101)
  rows <- c(2L, 5L, 9L, 12L)
  cols <- c(1L, 4L, 6L, 9L, 14L)
  repl <- matrix(runif(length(rows) * length(cols)), nrow = length(rows))

  x[rows, cols] <- repl
  mat[rows, cols] <- repl

  expect_equal(as.matrix(x), mat, tolerance = 1e-6)
})

test_that("subset assignment with mlx indices matches base R", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)

  rows <- as_mlx(c(1L, 3L))
  cols <- as_mlx(c(2L, 3L))
  block <- matrix(c(100, 200, 300, 400), nrow = 2)

  x[rows, cols] <- block
  mat[c(1, 3), c(2, 3)] <- block

  expect_equal(as.matrix(x), mat, tolerance = 1e-6)
})

test_that("3D boolean mask assignment preserves column-major ordering", {
  # Regression test for row-major vs column-major ordering
  arr <- array(1:24, dim = c(2, 3, 4))
  x <- as_mlx(arr)

  mask_dim1 <- c(TRUE, FALSE)
  mask_dim2 <- c(FALSE, TRUE, TRUE)
  mask_dim3 <- c(TRUE, FALSE, TRUE, FALSE)

  # Use distinct ordered values to detect ordering bugs
  values <- 101:104

  # Expected R column-major order: (1,2,1), (1,3,1), (1,2,3), (1,3,3)
  x_mlx <- as_mlx(arr)
  mask1_mlx <- as_mlx(mask_dim1, dtype = "bool")
  mask2_mlx <- as_mlx(mask_dim2, dtype = "bool")
  mask3_mlx <- as_mlx(mask_dim3, dtype = "bool")
  x_mlx[mask1_mlx, mask2_mlx, mask3_mlx] <- values

  arr_r <- arr
  arr_r[mask_dim1, mask_dim2, mask_dim3] <- values

  expect_equal(as.array(x_mlx), arr_r, tolerance = 1e-6)

  # Verify specific positions to ensure ordering is correct
  result <- as.array(x_mlx)
  expect_equal(result[1, 2, 1], 101)
  expect_equal(result[1, 3, 1], 102)
  expect_equal(result[1, 2, 3], 103)
  expect_equal(result[1, 3, 3], 104)
})

test_that("boolean mask assignment validates recycling rules", {
  # R requires n_selected to be a multiple of value length
  x <- mlx_matrix(rep(0, 16), 4, 4)
  mask1 <- as_mlx(c(TRUE, TRUE, TRUE, FALSE), dtype = "bool")
  mask2 <- as_mlx(c(TRUE, TRUE, TRUE, FALSE), dtype = "bool")

  # 9 selected elements
  # Valid: scalar (always works)
  x[mask1, mask2] <- 99
  expect_equal(as.array(x)[1:3, 1:3], matrix(99, 3, 3))

  # Valid: 3 elements, 9 = 3 * 3
  x[mask1, mask2] <- mlx_vector(1:3)
  result <- as.array(x)
  expect_equal(result[1:3, 1:3], matrix(rep(1:3, each = 3), 3, 3, byrow = TRUE))

  # Valid: 9 elements, exact match
  x[mask1, mask2] <- mlx_vector(101:109)
  result <- as.array(x)
  expect_equal(result[1, 1], 101)
  expect_equal(result[3, 3], 109)

  # Invalid: 5 elements, 9 is not a multiple of 5
  expect_error(
    x[mask1, mask2] <- mlx_vector(1:5),
    "number of items to replace is not a multiple of replacement length"
  )

  # Invalid: 2 elements, 9 is not a multiple of 2
  expect_error(
    x[mask1, mask2] <- mlx_vector(1:2),
    "number of items to replace is not a multiple of replacement length"
  )
})

test_that("subset assignment handles irregular numeric axes", {
  set.seed(20251115)
  arr <- array(runif(4 * 5 * 6), dim = c(4, 5, 6))
  x <- as_mlx(arr)

  rows <- c(4L, 1L)
  cols <- c(5L, 2L, 4L)
  slabs <- c(6L, 1L, 3L)
  repl <- array(runif(length(rows) * length(cols) * length(slabs)),
                dim = c(length(rows), length(cols), length(slabs)))

  x[rows, cols, slabs] <- repl
  arr[rows, cols, slabs] <- repl

  expect_equal(as.array(x), arr, tolerance = 1e-6)
})

test_that("subset assignment handles irregular mlx indices with missing axes", {
  set.seed(20251116)
  arr <- array(runif(3 * 7 * 5), dim = c(3, 7, 5))
  x <- as_mlx(arr)

  rows_vec <- c(3L, 1L)
  slabs_vec <- c(5L, 2L)
  rows <- as_mlx(rows_vec)
  slabs <- as_mlx(slabs_vec)
  repl <- array(runif(length(rows_vec) * dim(arr)[2] * length(slabs_vec)),
                dim = c(length(rows_vec), dim(arr)[2], length(slabs_vec)))

  x[rows, , slabs] <- repl
  arr[rows_vec, , slabs_vec] <- repl

  expect_equal(as.array(x), arr, tolerance = 1e-6)
})

test_that("subset assignment handles repeated numeric indices", {
  set.seed(20251101)
  mat <- matrix(seq_len(100), nrow = 10, ncol = 10)
  x <- as_mlx(mat)

  rows <- c(1L, 3L, 3L)
  cols <- c(4L, 2L, 4L)
  values <- matrix(runif(length(rows) * length(cols)), nrow = length(rows))

  x[rows, cols] <- values
  mat[rows, cols] <- values

  expect_equal(as.matrix(x), mat, tolerance = 1e-6)
})

test_that("subset assignment preserves order of numeric indices", {
  set.seed(20251102)
  mat <- matrix(seq_len(100), nrow = 10, ncol = 10)
  x <- as_mlx(mat)

  rows <- c(5L, 2L)
  cols <- c(7L, 1L, 4L)
  values <- matrix(runif(length(rows) * length(cols)), nrow = length(rows))

  x[rows, cols] <- values
  mat[rows, cols] <- values

  expect_equal(as.matrix(x), mat, tolerance = 1e-6)
})

test_that("mlx matrix assignment works", {
  mat <- matrix(1:12, 3, 4)
  x <- as_mlx(mat)

  idx <- mlx_matrix(c(1, 1,
                         3, 4), ncol = 2, byrow = TRUE)
  vals <- c(500, 600)

  x[idx] <- vals
  mat[matrix(c(1, 1, 3, 4), ncol = 2, byrow = TRUE)] <- vals

  expect_equal(as.matrix(x), mat, tolerance = 1e-6)
})

test_that("mlx matrix assignment with duplicates keeps last value", {
  mat <- matrix(0, 3, 3)
  x <- as_mlx(mat)

  idx <- mlx_matrix(c(1, 1,
                         1, 1,
                         2, 2), ncol = 2, byrow = TRUE)
  vals <- c(5, 7, 9)

  x[idx] <- vals
  mat[matrix(c(1, 1, 1, 1, 2, 2), ncol = 2, byrow = TRUE)] <- vals

  expect_equal(as.matrix(x), mat, tolerance = 1e-6)
})

test_that("negative numeric indices work for assignment", {
  vec <- 1:5
  mlx_vec <- as_mlx(vec)

  mlx_vec[-1] <- 0
  vec[-1] <- 0

  expect_equal(as.vector(mlx_vec), vec)
})

test_that("subset assignment preserves GPU device", {
  skip_if_not(mlx_has_gpu())
  mat <- matrix(1:6, 2, 3)
  x <- as_mlx(mat, device = "gpu")

  x[1, ] <- c(10, 20, 30)
  mat[1, ] <- c(10, 20, 30)

  expect_equal(x$device, "gpu")
  expect_equal(as.matrix(x), mat, tolerance = 1e-6)
})

test_that("subset assignment: mlx numeric positive indices", {
  # 1D vector
  x <- mlx_vector(1:10)
  idx_mlx <- as_mlx(c(2L, 4L, 6L))
  x[idx_mlx] <- 99
  expect_equal(as.vector(x)[c(2, 4, 6)], rep(99, 3))

  # 2D matrix
  x <- mlx_matrix(1:20, 4, 5)
  idx_mlx <- as_mlx(c(1L, 3L))
  x[idx_mlx, ] <- 88
  result <- as.matrix(x)
  expect_equal(result[c(1, 3), ], matrix(88, 2, 5))

  # 3D array
  x <- mlx_array(1:60, c(3, 4, 5))
  idx_mlx <- as_mlx(c(1L, 3L))
  x[idx_mlx, , ] <- 77
  result <- as.array(x)
  expect_equal(result[c(1, 3), , ], array(77, c(2, 4, 5)))
})

test_that("subset assignment: mlx numeric negative indices", {
  # 1D vector
  x <- mlx_vector(1:10)
  idx_mlx <- as_mlx(c(-2L, -4L, -6L))
  x[idx_mlx] <- 99
  result <- as.vector(x)
  expect_equal(result[c(-2, -4, -6)], rep(99, 7))

  # 2D matrix
  x <- mlx_matrix(1:20, 4, 5)
  idx_mlx <- as_mlx(c(-1L, -3L))
  x[idx_mlx, ] <- 88
  result <- as.matrix(x)
  expect_equal(result[-c(1, 3), ], matrix(88, 2, 5))

  # 3D array
  x <- mlx_array(1:60, c(3, 4, 5))
  idx_mlx <- as_mlx(-2L)
  x[idx_mlx, , ] <- 77
  result <- as.array(x)
  expect_equal(result[-2, , ], array(77, c(2, 4, 5)))
})

test_that("subset assignment: R numeric positive indices", {
  # 1D vector
  x <- mlx_vector(1:10)
  x[c(2, 4, 6)] <- 99
  expect_equal(as.vector(x)[c(2, 4, 6)], rep(99, 3))

  # 2D matrix
  x <- mlx_matrix(1:20, 4, 5)
  x[c(1, 3), ] <- 88
  result <- as.matrix(x)
  expect_equal(result[c(1, 3), ], matrix(88, 2, 5))

  # 3D array
  x <- mlx_array(1:60, c(3, 4, 5))
  x[c(1, 3), , ] <- 77
  result <- as.array(x)
  expect_equal(result[c(1, 3), , ], array(77, c(2, 4, 5)))
})

test_that("subset assignment: R numeric negative indices", {
  # 1D vector
  x <- mlx_vector(1:10)
  x[c(-2, -4, -6)] <- 99
  result <- as.vector(x)
  expect_equal(result[c(-2, -4, -6)], rep(99, 7))

  # 2D matrix
  x <- mlx_matrix(1:20, 4, 5)
  x[-c(1, 3), ] <- 88
  result <- as.matrix(x)
  expect_equal(result[-c(1, 3), ], matrix(88, 2, 5))

  # 3D array
  x <- mlx_array(1:60, c(3, 4, 5))
  x[-2, , ] <- 77
  result <- as.array(x)
  expect_equal(result[-2, , ], array(77, c(2, 4, 5)))
})

test_that("subset assignment: mlx boolean masks", {
  # 1D vector
  x <- mlx_vector(1:10)
  mask <- as_mlx(c(TRUE, FALSE, TRUE, FALSE, TRUE, rep(FALSE, 5)), dtype = "bool")
  x[mask] <- 99
  expect_equal(as.vector(x)[c(1, 3, 5)], rep(99, 3))

  # 2D matrix
  x <- mlx_matrix(1:20, 4, 5)
  mask <- as_mlx(c(TRUE, FALSE, TRUE, FALSE), dtype = "bool")
  x[mask, ] <- 88
  result <- as.matrix(x)
  expect_equal(result[c(1, 3), ], matrix(88, 2, 5))

  # 3D array
  x <- mlx_array(1:60, c(3, 4, 5))
  mask <- as_mlx(c(TRUE, FALSE, TRUE), dtype = "bool")
  x[mask, , ] <- 77
  result <- as.array(x)
  expect_equal(result[c(1, 3), , ], array(77, c(2, 4, 5)))
})

test_that("subset assignment: R logical masks", {
  # 1D vector
  x <- mlx_vector(1:10)
  x[c(TRUE, FALSE, TRUE, FALSE, TRUE, rep(FALSE, 5))] <- 99
  expect_equal(as.vector(x)[c(1, 3, 5)], rep(99, 3))

  # 2D matrix
  x <- mlx_matrix(1:20, 4, 5)
  x[c(TRUE, FALSE, TRUE, FALSE), ] <- 88
  result <- as.matrix(x)
  expect_equal(result[c(1, 3), ], matrix(88, 2, 5))

  # 3D array
  x <- mlx_array(1:60, c(3, 4, 5))
  x[c(TRUE, FALSE, TRUE), , ] <- 77
  result <- as.array(x)
  expect_equal(result[c(1, 3), , ], array(77, c(2, 4, 5)))
})

test_that("subset assignment: mixed index types", {
  # mlx boolean + R numeric
  x <- mlx_matrix(1:20, 4, 5)
  mask <- as_mlx(c(TRUE, FALSE, TRUE, FALSE), dtype = "bool")
  x[mask, c(2, 4)] <- 99
  result <- as.matrix(x)
  expect_equal(result[c(1, 3), c(2, 4)], matrix(99, 2, 2))

  # R logical + mlx numeric
  x <- mlx_matrix(1:20, 4, 5)
  idx_mlx <- as_mlx(c(2L, 4L))
  x[c(TRUE, FALSE, TRUE, FALSE), idx_mlx] <- 88
  result <- as.matrix(x)
  expect_equal(result[c(1, 3), c(2, 4)], matrix(88, 2, 2))

  # mlx numeric + mlx boolean on 3D
  x <- mlx_array(1:60, c(3, 4, 5))
  idx_mlx <- as_mlx(c(1L, 3L))
  mask <- as_mlx(c(TRUE, FALSE, TRUE, FALSE), dtype = "bool")
  x[idx_mlx, mask, ] <- 77
  result <- as.array(x)
  expect_equal(result[c(1, 3), c(1, 3), ], array(77, c(2, 2, 5)))

  # R numeric negative + mlx boolean
  x <- mlx_matrix(1:20, 4, 5)
  mask <- as_mlx(c(TRUE, FALSE, TRUE, FALSE, TRUE), dtype = "bool")
  x[-c(1, 4), mask] <- 66
  result <- as.matrix(x)
  expect_equal(result[c(2, 3), c(1, 3, 5)], matrix(66, 2, 3))
})
