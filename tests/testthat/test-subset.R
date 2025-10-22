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
