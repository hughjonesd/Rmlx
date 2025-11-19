test_that("basic slicing matches base semantics", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)

  expect_equal(as.matrix(x[1, ]), matrix(mat[1, ], nrow = 1, byrow = TRUE))
  expect_equal(as.vector(x[1, , drop = TRUE]), mat[1, ])

  expect_equal(as.matrix(x[, 2]), matrix(mat[, 2], ncol = 1))
  expect_equal(as.matrix(x[2, 3]), matrix(mat[2, 3], nrow = 1, ncol = 1))

  expect_equal(as.matrix(x[mat[, 1] > 1, ]), mat[mat[, 1] > 1, , drop = FALSE])
})

test_that("logical masks work", {
  mat <- matrix(1:6, 2, 3)
  mask_rows <- c(TRUE, FALSE)
  mask_cols <- c(TRUE, FALSE, TRUE)
  x <- as_mlx(mat)

  expect_equal(as.matrix(x[mask_rows, ]), mat[mask_rows, , drop = FALSE])
  expect_equal(as.matrix(x[, mask_cols]), mat[, mask_cols, drop = FALSE])
  expect_equal(as.matrix(x[FALSE, ]), mat[FALSE, , drop = FALSE])
})

test_that("single logical index flattens like base R", {
  base_mat <- matrix((-4):3, nrow = 2)
  mask <- base_mat < 0
  mat <- base_mat
  expected <- mat
  expected[mask] <- 0
  x <- as_mlx(mat)

  expect_equal(as.vector(x[mask]), mat[mask])

  x[mask] <- 0
  expect_equal(as.matrix(x), expected)

  mat <- base_mat
  mask_mlx <- as_mlx(mask)
  x_mlx_mask <- as_mlx(mat)
  expect_equal(as.vector(x_mlx_mask[mask_mlx]), mat[mask])
  x_mlx_mask[mask_mlx] <- 0
  expect_equal(as.matrix(x_mlx_mask), expected)
})

test_that("mlx logical masks work like R logical masks", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)

  # Test row indexing with mlx logical
  mask_rows <- c(TRUE, TRUE, FALSE)
  mask_rows_mlx <- as_mlx(mask_rows)
  expect_equal(as.matrix(x[mask_rows_mlx, ]), mat[mask_rows, , drop = FALSE])

  # Test column indexing with mlx logical
  mask_cols <- c(TRUE, FALSE, TRUE)
  mask_cols_mlx <- as_mlx(mask_cols)
  expect_equal(as.matrix(x[, mask_cols_mlx]), mat[, mask_cols, drop = FALSE])

  # Test both dimensions
  expect_equal(as.matrix(x[mask_rows_mlx, mask_cols_mlx]),
               mat[mask_rows, mask_cols, drop = FALSE])

  # Test all FALSE
  mask_false <- as_mlx(c(FALSE, FALSE, FALSE))
  expect_equal(as.matrix(x[mask_false, ]), mat[c(FALSE, FALSE, FALSE), , drop = FALSE])
})

test_that("higher dimensional indexing works", {
  arr <- array(seq_len(24), dim = c(3, 4, 2))
  x <- as_mlx(arr)

  expect_equal(as.array(x[1, , ]), arr[1, , , drop = FALSE])
  expect_equal(as.array(x[, 2, ]), arr[, 2, , drop = FALSE])
  expect_equal(as.array(x[, , 1]), arr[, , 1, drop = FALSE])

  expect_equal(as.array(x[2, c(1, 3), 2]), arr[2, c(1, 3), 2, drop = FALSE])
  expect_equal(as.array(x[, c(TRUE, FALSE, TRUE, FALSE), ]),
               arr[, c(TRUE, FALSE, TRUE, FALSE), , drop = FALSE])
})

test_that("drop argument matches expectations", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)

  res <- x[1, ]
  expect_equal(dim(res), c(1L, ncol(mat)))

  res_drop <- x[1, , drop = TRUE]
  expect_equal(as.vector(res_drop), mat[1, ])
})

test_that("zero length selections return empty tensors", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)

  empty_rows <- x[FALSE, ]
  expect_equal(dim(empty_rows), c(0L, ncol(mat)))

  empty_cols <- x[, integer(0)]
  expect_equal(dim(empty_cols), c(nrow(mat), 0L))
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

test_that("matrix indexing matches base behaviour", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)

  idx <- cbind(c(1, 3), c(2, 3))
  expect_equal(as.vector(x[idx]), mat[idx], tolerance = 1e-6)
})

test_that("subset handles repeated numeric indices", {
  mat <- matrix(seq_len(100), nrow = 10, ncol = 10)
  x <- as_mlx(mat)

  rows <- c(1L, 3L, 3L)
  cols <- c(4L, 2L, 4L)

  expect_equal(as.matrix(x[rows, cols]), mat[rows, cols, drop = FALSE])
})

test_that("subset preserves the order of numeric indices", {
  mat <- matrix(seq_len(100), nrow = 10, ncol = 10)
  x <- as_mlx(mat)

  rows <- c(5L, 2L)
  cols <- c(7L, 1L, 4L)

  expect_equal(as.matrix(x[rows, cols]), mat[rows, cols, drop = FALSE])
})

test_that("direct gather and slice_update mirror MLX semantics", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)

  gathered <- mlx_gather(x, list(c(1L, 3L)), axes = 1L)
  expect_equal(as.matrix(gathered), mat[c(1, 3), , drop = FALSE], tolerance = 1e-6)

  updated <- mlx_slice_update(
    x,
    as_mlx(matrix(c(100, 200, 300, 400), nrow = 2)),
    start = c(1L, 2L),
    stop = c(2L, 3L),
    strides = c(1L, 1L)
  )
  mat[1:2, 2:3] <- matrix(c(100, 200, 300, 400), nrow = 2)
  expect_equal(as.matrix(updated), mat, tolerance = 1e-6)
})

test_that("mlx_gather treats negative indices as omissions", {
  x <- as_mlx(1:10)

  dropped <- mlx_gather(x, list(c(-1L, -2L)))
  expect_equal(as.vector(dropped), 3:10)

  dropped_mlx <- mlx_gather(x, list(as_mlx(c(-1L, -2L))))
  expect_equal(as.vector(dropped_mlx), 3:10)
})

test_that("mlx_gather supports multi-axis tensors", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)

  idx_rows <- matrix(c(1L, 2L, 3L, 1L), nrow = 2, byrow = TRUE)
  idx_cols <- matrix(c(1L, 3L, 2L, 2L), nrow = 2, byrow = TRUE)

  gathered <- mlx_gather(x, list(idx_rows, idx_cols), axes = c(1L, 2L))
  expected <- array(mat[cbind(as.vector(idx_rows), as.vector(idx_cols))], dim(idx_rows))

  expect_equal(as.array(gathered), expected, tolerance = 1e-6)
})

test_that("mlx_gather preserves remaining axes and errors on invalid axes", {
  arr <- array(seq_len(24), dim = c(4, 3, 2))
  x <- as_mlx(arr)

  idx_rows <- matrix(c(1L, 4L, 2L, 3L), nrow = 2, byrow = TRUE)
  idx_cols <- as_mlx(matrix(c(2L, 1L, 3L, 2L), nrow = 2, byrow = TRUE))
  idx_cols_mat <- as.matrix(idx_cols)

  gathered <- mlx_gather(x, list(idx_rows, idx_cols), axes = c(1L, 2L))

  expected <- array(0, dim = c(dim(idx_rows), dim(arr)[3]))
  for (i in seq_len(dim(idx_rows)[1])) {
    for (j in seq_len(dim(idx_rows)[2])) {
      expected[i, j, ] <- arr[idx_rows[i, j], idx_cols_mat[i, j], ]
    }
  }

  expect_equal(as.array(gathered), expected, tolerance = 1e-6)
  expect_error(mlx_gather(x, list(1L), axes = -1L), "Negative axes")
  expect_error(mlx_gather(x, list(1L, 2L), axes = c(1L, 1L)), "must not contain duplicates")
})

test_that("mlx vector indexing works (1-based)", {
  mat <- matrix(1:12, 3, 4)
  x <- as_mlx(mat)

  # Test row indexing with mlx vector
  idx_rows <- as_mlx(c(1L, 3L))
  result <- x[idx_rows, ]
  expect_equal(as.matrix(result), mat[c(1, 3), , drop = FALSE])

  # Test column indexing with mlx vector
  idx_cols <- as_mlx(c(2L, 4L))
  result <- x[, idx_cols]
  expect_equal(as.matrix(result), mat[, c(2, 4), drop = FALSE])

  # Test both dimensions with mlx vectors
  result <- x[idx_rows, idx_cols]
  expect_equal(as.matrix(result), mat[c(1, 3), c(2, 4), drop = FALSE])

  # Test single row with mlx scalar
  idx_single <- as_mlx(2L)
  result <- x[idx_single, ]
  expect_equal(as.matrix(result), matrix(mat[2, ], nrow = 1))
})

test_that("mlx vector indexing uses 1-based convention", {
  vec <- as_mlx(c(10, 20, 30, 40, 50))

  # Index 1 should get first element (10), not second
  idx <- as_mlx(1L)
  expect_equal(as.vector(vec[idx]), 10)

  # Index 5 should get last element (50)
  idx <- as_mlx(5L)
  expect_equal(as.vector(vec[idx]), 50)

  # Multiple indices
  idx <- as_mlx(c(2L, 4L))
  expect_equal(as.vector(vec[idx]), c(20, 40))
})

test_that("mlx vector indexing handles automatic dtype conversion", {
  mat <- matrix(1:12, 3, 4)
  x <- as_mlx(mat)

  # mlx arrays from R integers default to float32, should be auto-converted
  idx <- as_mlx(c(1L, 3L))  # This will be float32
  expect_equal(mlx_dtype(idx), "float32")

  # But indexing should still work (auto-converts to int64)
  result <- x[idx, ]
  expect_equal(as.matrix(result), mat[c(1, 3), , drop = FALSE])

  # Explicit integer dtype should also work
  idx_int <- as_mlx(c(1L, 3L), dtype = "int32")
  result <- x[idx_int, ]
  expect_equal(as.matrix(result), mat[c(1, 3), , drop = FALSE])
})

test_that("mlx matrix indexing works (1-based)", {
  mat <- matrix(1:12, 3, 4)
  x <- as_mlx(mat)

  # Test matrix-style indexing (each row is [row, col])
  idx_mat <- matrix(c(1, 1,
                      2, 2,
                      3, 3), ncol = 2, byrow = TRUE)

  # R version (baseline)
  expected <- mat[idx_mat]

  # mlx version
  idx_mat_mlx <- as_mlx(idx_mat)
  result <- x[idx_mat_mlx]

  expect_equal(as.vector(result), expected)
})

test_that("mlx matrix indexing extracts specific elements", {
  mat <- matrix(1:12, 3, 4)
  x <- as_mlx(mat)

  # Extract diagonal elements
  idx_diag <- matrix(c(1, 1,
                       2, 2,
                       3, 3), ncol = 2, byrow = TRUE)
  idx_diag_mlx <- as_mlx(idx_diag)

  result <- x[idx_diag_mlx]
  expect_equal(as.vector(result), c(1, 5, 9))

  # Extract corner elements
  idx_corners <- matrix(c(1, 1,
                          1, 4,
                          3, 1,
                          3, 4), ncol = 2, byrow = TRUE)
  idx_corners_mlx <- as_mlx(idx_corners)

  result <- x[idx_corners_mlx]
  expect_equal(as.vector(result), c(1, 10, 3, 12))
})

test_that("mlx matrix indexing uses 1-based convention", {
  mat <- matrix(1:12, 3, 4)
  x <- as_mlx(mat)

  # [1, 1] should get first element (1), not [0, 0]
  idx <- as_mlx(matrix(c(1, 1), nrow = 1))
  expect_equal(as.vector(x[idx]), 1)

  # [3, 4] should get last element (12)
  idx <- as_mlx(matrix(c(3, 4), nrow = 1))
  expect_equal(as.vector(x[idx]), 12)
})

test_that("mlx matrix assignment works", {
  mat <- matrix(1:12, 3, 4)
  x <- as_mlx(mat)

  idx <- as_mlx(matrix(c(1, 1,
                         3, 4), ncol = 2, byrow = TRUE))
  vals <- c(500, 600)

  x[idx] <- vals
  mat[matrix(c(1, 1, 3, 4), ncol = 2, byrow = TRUE)] <- vals

  expect_equal(as.matrix(x), mat, tolerance = 1e-6)
})

test_that("mlx matrix assignment with duplicates keeps last value", {
  mat <- matrix(0, 3, 3)
  x <- as_mlx(mat)

  idx <- as_mlx(matrix(c(1, 1,
                         1, 1,
                         2, 2), ncol = 2, byrow = TRUE))
  vals <- c(5, 7, 9)

  x[idx] <- vals
  mat[matrix(c(1, 1, 1, 1, 2, 2), ncol = 2, byrow = TRUE)] <- vals

  expect_equal(as.matrix(x), mat, tolerance = 1e-6)
})

test_that("negative numeric indices behave like base R", {
  vec <- 1:5
  mlx_vec <- as_mlx(vec)

  expect_equal(as.vector(mlx_vec[-1]), vec[-1])
  expect_equal(as.vector(mlx_vec[-c(1, 3)]), vec[-c(1, 3)])
  expect_equal(length(mlx_vec[integer(0)]), 0L)
  expect_equal(as.vector(mlx_vec[-integer(0)]), vec[-integer(0)])

  expect_error(mlx_vec[c(-1, 2)], "Cannot mix positive and negative indices", fixed = TRUE)
  expect_error(mlx_vec[c(-1, 0)], "Index contains zeros", fixed = TRUE)
})

test_that("negative numeric indices work for assignment", {
  vec <- 1:5
  mlx_vec <- as_mlx(vec)

  mlx_vec[-1] <- 0
  vec[-1] <- 0

  expect_equal(as.vector(mlx_vec), vec)
})

test_that("negative indices mix with logical axes", {
  mat <- matrix(1:25, 5)
  x <- as_mlx(mat)

  expect_equal(
    as.matrix(x[c(TRUE, FALSE, TRUE, FALSE, TRUE), -(1:2)]),
    mat[c(TRUE, FALSE, TRUE, FALSE, TRUE), -(1:2), drop = FALSE]
  )
})

test_that("negative matrix indices are rejected", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)
  neg_idx <- cbind(c(-1, -2), c(1, 2))

  expect_error(x[neg_idx], "Matrix indices must be positive", fixed = TRUE)
  expect_error(x[as_mlx(neg_idx)], "Matrix indices must be positive", fixed = TRUE)
})

test_that("mlx and R indexing give identical results", {
  set.seed(123)
  mat <- matrix(rnorm(20), 5, 4)
  x <- as_mlx(mat)

  # Test 1: Vector indexing
  idx_rows <- c(1L, 3L, 5L)
  idx_mlx_rows <- as_mlx(idx_rows)
  idx_cols <- c(1L, 3L, 4L)
  idx_mlx_cols <- as_mlx(idx_cols)

  expect_equal(as.matrix(x[idx_rows, ]), as.matrix(x[idx_mlx_rows, ]))
  expect_equal(as.matrix(x[, idx_cols]), as.matrix(x[, idx_mlx_cols]))

  # Test 2: Matrix indexing
  idx_mat <- matrix(c(1, 1,
                      2, 3,
                      4, 2,
                      5, 4), ncol = 2, byrow = TRUE)
  idx_mat_mlx <- as_mlx(idx_mat)

  expect_equal(as.vector(x[idx_mat]),
               as.vector(x[idx_mat_mlx]))
})

test_that("mlx indexing works with higher dimensional arrays", {
  arr <- array(1:24, dim = c(3, 4, 2))
  x <- as_mlx(arr)

  # Test vector indexing in 3D
  idx <- as_mlx(c(1L, 3L))
  result <- x[idx, , ]
  expect_equal(as.array(result), arr[c(1, 3), , , drop = FALSE])

  # Test multiple dimensions
  idx1 <- as_mlx(c(1L, 2L))
  idx2 <- as_mlx(c(2L, 4L))
  result <- x[idx1, idx2, ]
  expect_equal(as.array(result), arr[c(1, 2), c(2, 4), , drop = FALSE])
})

test_that("mlx indexing preserves device", {
  mat <- matrix(1:12, 3, 4)
  x <- as_mlx(mat, device = "gpu")

  idx <- as_mlx(c(1L, 3L), device = "gpu")
  result <- x[idx, ]

  expect_equal(result$device, "gpu")
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

test_that("empty mlx index returns empty result", {
  mat <- matrix(1:12, 3, 4)
  x <- as_mlx(mat)

  # Empty vector index
  idx_empty <- as_mlx(integer(0), dtype = "int32")
  result <- x[idx_empty, ]

  expect_equal(nrow(result), 0L)
  expect_equal(ncol(result), ncol(mat))
})

test_that("mlx indexing errors appropriately", {
  mat <- matrix(1:12, 3, 4)
  x <- as_mlx(mat)

  # Note: MLX does not error on out-of-bounds indices with mlx arrays
  # It may return zeros or undefined values. This differs from R's behavior
  # but is consistent with MLX's C++ API and lazy evaluation model.
  # We only get bounds checking when using R integer vectors:
  expect_error(x[c(1L, 10L), ], "Index out of bounds")

  # Matrix with wrong number of columns should error
  idx_mat_wrong <- as_mlx(matrix(c(1, 1, 1), nrow = 1))  # 3 columns for 2D array
  expect_error(x[idx_mat_wrong], "one column per dimension")
})
