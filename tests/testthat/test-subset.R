test_that("basic slicing matches base semantics", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)

  expect_equal(as.matrix(x[1, ]), matrix(mat[1, ], nrow = 1, byrow = TRUE))
  expect_equal(as.vector(as.matrix(x[1, , drop = TRUE])), mat[1, ])

  expect_equal(as.matrix(x[, 2]), matrix(mat[, 2], ncol = 1))
  expect_equal(as.matrix(x[2, 3]), matrix(mat[2, 3], nrow = 1, ncol = 1))

  expect_equal(as.matrix(x[-1, ]), mat[-1, , drop = FALSE])
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
  expect_equal(as.vector(as.matrix(res_drop)), mat[1, ])
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

test_that("subset assignment with logical masks behaves like base R", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)

  row_mask <- c(TRUE, FALSE, TRUE)
  col_mask <- c(FALSE, TRUE, TRUE)

  x[row_mask, col_mask] <- c(5, 6, 7, 8)
  mat[row_mask, col_mask] <- c(5, 6, 7, 8)

  expect_equal(as.matrix(x), mat, tolerance = 1e-6)
})

test_that("matrix indexing matches base behaviour", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)

  idx <- cbind(c(1, 3), c(2, 3))
  expect_equal(as.vector(as.matrix(x[idx])), mat[idx], tolerance = 1e-6)
})

test_that("direct gather and slice_update mirror MLX semantics", {
  mat <- matrix(1:9, 3, 3)
  x <- as_mlx(mat)

  gathered <- mlx_gather(x, list(c(1L, 3L)), axes = 1L)
  expect_equal(as.matrix(gathered), mat[c(1, 3), , drop = FALSE], tolerance = 1e-6)

  updated <- mlx_slice_update(
    x,
    as_mlx(matrix(c(100, 200, 300, 400), nrow = 2)),
    start = c(0L, 1L),
    stop = c(2L, 3L),
    strides = c(1L, 1L)
  )
  mat[1:2, 2:3] <- matrix(c(100, 200, 300, 400), nrow = 2)
  expect_equal(as.matrix(updated), mat, tolerance = 1e-6)
})
