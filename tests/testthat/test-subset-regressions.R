test_that("short logical indices recycle for R and mlx indices", {
  mat <- matrix(1:24, 6, 4)
  for (convert in list(identity, as_mlx)) {
    x <- as_mlx(mat)
    expect_equal(as.matrix(x[convert(TRUE), ]), mat)
    expect_equal(as.matrix(x[convert(c(TRUE, FALSE)), convert(TRUE)]),
                 mat[c(TRUE, FALSE), , drop = FALSE])
    x[convert(c(TRUE, FALSE)), convert(TRUE)] <- 101:104
    expected <- mat
    expected[c(TRUE, FALSE), ] <- 101:104
    expect_equal(as.matrix(x), expected)
  }
})

test_that("single-index reads require flattening except for coordinate matrices", {
  x <- mlx_matrix(1:12, 3, 4)
  expect_error(x[1:2], "flatten")
  expect_error(x[x > 3], "flatten")
  expect_equal(as.matrix(x[]), matrix(1:12, 3, 4))
  expect_equal(as.vector(mlx_flatten(x)[1:2]), c(1, 4))
})

test_that("coordinate assignment accepts R and mlx matrices and recycles in R order", {
  mat <- matrix(1:12, 3, 4)
  idx <- cbind(c(1, 2, 3, 1), c(1, 2, 3, 4))
  for (convert in list(identity, as_mlx)) {
    x <- as_mlx(mat)
    expected <- mat
    x[convert(idx)] <- c(100, 200)
    expected[idx] <- c(100, 200)
    expect_equal(as.matrix(x), expected)
    x[convert(idx)] <- as_mlx(matrix(101:104, 2))
    expected[idx] <- matrix(101:104, 2)
    expect_equal(as.matrix(x), expected)
  }
})

test_that("linear assignment uses R column-major order", {
  arr <- array(1:24, c(2, 3, 4))
  for (convert in list(identity, as_mlx)) {
    x <- as_mlx(arr)
    expected <- arr
    x[convert(c(1L, 4L, 17L, 24L))] <- 101:104
    expected[c(1L, 4L, 17L, 24L)] <- 101:104
    expect_equal(as.array(x), expected)
    mask <- arr %% 3 == 0
    x[convert(mask)] <- 201:204
    expected[mask] <- 201:204
    expect_equal(as.array(x), expected)
  }
})

test_that("coordinate indices validate bounds in each dimension", {
  x <- mlx_matrix(1:12, 3, 4)
  for (convert in list(identity, as_mlx)) {
    expect_error(x[convert(matrix(c(4, 1), nrow = 1))], "out of bounds")
    expect_error(x[convert(matrix(c(1, 5), nrow = 1))] <- 0, "out of bounds")
  }
})

test_that("explicit NULL and empty selections preserve empty dimensions and names", {
  mat <- matrix(1:12, 3, 4, dimnames = list(letters[1:3], LETTERS[1:4]))
  x <- as_mlx(mat)
  expect_equal(as.matrix(x[NULL, ]), mat[NULL, , drop = FALSE])
  expect_equal(as.matrix(x[integer(0), 1:2]), mat[integer(0), 1:2, drop = FALSE])
  x[integer(0), ] <- numeric(0)
  expect_equal(as.matrix(x), mat)
  x[NULL, ] <- 100
  expect_equal(as.matrix(x), mat)
})

for (device in c("cpu", "gpu")) {
  with_device(device, {
    test_that(paste("rectangular subsets and replacements match R on", device), {
      mat <- matrix(1:20, 4, 5, dimnames = list(rows = letters[1:4], cols = LETTERS[1:5]))
      row_indices <- list(1:4, 2:3, 4:1, c(4L, 1L), integer(0), -2L, c(TRUE, FALSE))
      col_indices <- list(1:5, 2:4, 5:1, c(5L, 2L), integer(0), -c(1L, 3L))
      for (convert in list(identity, as_mlx)) {
        for (i in row_indices) {
          for (j in col_indices) {
            x <- as_mlx(mat)
            expected <- mat[i, j, drop = FALSE]
            expect_equal(as.matrix(x[convert(i), convert(j)]), expected)
            values <- if (length(expected) %% 2L == 0L) c(101, 102) else 101
            x[convert(i), convert(j)] <- values
            mat_updated <- mat
            mat_updated[i, j] <- values
            expect_equal(as.matrix(x), mat_updated)
          }
        }
      }
    })

    test_that(paste("slices and sparse subsets preserve array dtype on", device), {
      for (dtype in c("float32", "int32", "bool")) {
        arr <- array(1:60, c(3, 4, 5))
        if (dtype == "bool") arr <- arr %% 3 == 0
        x <- as_mlx(arr, dtype = dtype)
        expect_equal(as.array(x[2:3, 2:4, 2:5]), arr[2:3, 2:4, 2:5, drop = FALSE])
        expect_equal(mlx_dtype(x[2:3, 2:4, 2:5]), dtype)
        expect_equal(as.array(x[c(3L, 1L), c(4L, 2L), c(5L, 1L)]),
                     arr[c(3L, 1L), c(4L, 2L), c(5L, 1L), drop = FALSE])
        expect_equal(mlx_dtype(x[c(3L, 1L), c(4L, 2L), c(5L, 1L)]), dtype)
        x[2:3, 2:4, 2:5] <- TRUE
        arr[2:3, 2:4, 2:5] <- TRUE
        expect_equal(as.array(x), arr)
        expect_equal(mlx_dtype(x), dtype)
        x[c(3L, 1L), c(4L, 2L), c(5L, 1L)] <- FALSE
        arr[c(3L, 1L), c(4L, 2L), c(5L, 1L)] <- FALSE
        expect_equal(as.array(x), arr)
        expect_equal(mlx_dtype(x), dtype)
      }
    })
  })
}

test_that("full reads and replacements preserve names and aliases", {
  arr <- array(1:24, c(2, 3, 4),
               dimnames = list(a = letters[1:2], b = LETTERS[1:3], c = as.character(1:4)))
  x <- as_mlx(arr)
  expect_equal(as.array(x[]), arr)
  expect_equal(as.array(x[, , ]), arr)
  expect_equal(as.array(x[1:2, 1:3, 1:4]), arr)
  expect_equal(as_r(x[1, , 1, drop = TRUE]), arr[1, , 1, drop = TRUE])
  expect_equal(as.array(x[1, integer(0), , drop = TRUE]),
               arr[1, integer(0), , drop = TRUE])
  alias <- x
  x[] <- as_mlx(array(101:124, c(4, 3, 2)))
  arr[] <- array(101:124, c(4, 3, 2))
  expect_equal(as.array(x), arr)
  expect_equal(as.vector(alias), 1:24)
  expect_equal(dimnames(x), dimnames(arr))
  x[, , ] <- c(10, 20)
  arr[, , ] <- c(10, 20)
  expect_equal(as.array(x), arr)

  v <- as_mlx(c(a = 1, b = 2, c = 3, d = 4))
  expect_equal(as_r(v[]), c(a = 1, b = 2, c = 3, d = 4))
  v[2:3] <- c(20, 30)
  expect_equal(as_r(v), c(a = 1, b = 20, c = 30, d = 4))
})

test_that("coordinate matrices cover empty, repeated, and higher rank reads", {
  arr <- array(1:24, c(2, 3, 4))
  idx <- rbind(c(2, 3, 4), c(1, 2, 3), c(2, 3, 4))
  for (convert in list(identity, as_mlx)) {
    x <- as_mlx(arr)
    expect_equal(as.vector(x[convert(idx)]), arr[idx])
    expect_null(dim(x[convert(idx)]))
    expect_equal(length(x[convert(matrix(integer(0), ncol = 3))]), 0L)
    x[convert(matrix(integer(0), ncol = 3))] <- numeric(0)
    expect_equal(as.array(x), arr)
    expect_error(x[convert(idx)] <- 0, "duplicate")
  }
})

test_that("all index paths reject invalid inputs and replacement lengths", {
  x <- mlx_matrix(1:12, 3, 4)
  for (convert in list(identity, as_mlx)) {
    for (idx in list(0L, c(1L, NA), c(-1L, 2L), 4L, -4L,
                     c(TRUE, FALSE, FALSE, FALSE))) {
      expect_error(x[convert(idx), ])
      expect_error(x[convert(idx), ] <- 0)
    }
    for (idx in list(matrix(c(0L, 1L), 1), matrix(c(NA, 1L), 1),
                     matrix(c(-1L, 1L), 1), matrix(1L, 1, 3), array(1L, c(1, 1, 2)))) {
      expect_error(x[convert(idx)])
      expect_error(x[convert(idx)] <- 0)
    }
    expect_error(x[convert(1:2), convert(1:2)] <- 1:3, "multiple")
    expect_error(x[convert(1:2), convert(1:2)] <- numeric(0), "length >= 1")
  }
  expect_error(x[1, 1, 1], "number of indices")
  expect_error(x[1, 1, 1] <- 0, "number of indices")
  expect_error(x[stop("missing value in custom index"), ], "missing value in custom index")
  expect_error(x[, stop("missing value in custom index")] <- 0,
               "missing value in custom index")
})

test_that("fractional numeric positions truncate before validation", {
  mat <- matrix(1:12, 3, 4)
  for (convert in list(identity, as_mlx)) {
    x <- as_mlx(mat)
    i <- c(1.9, 3.9)
    j <- c(4.8, 2.2)
    expect_equal(as.matrix(x[convert(i), convert(j)]), mat[i, j, drop = FALSE])
    expect_equal(as.matrix(x[convert(-1.9), ]), mat[-1.9, , drop = FALSE])
    x[convert(i), convert(j)] <- 101:104
    expected <- mat
    expected[i, j] <- 101:104
    expect_equal(as.matrix(x), expected)
    idx <- cbind(i, j)
    expect_equal(as.vector(x[convert(idx)]), expected[idx])
    x[convert(idx)] <- 201:202
    expected[idx] <- 201:202
    expect_equal(as.matrix(x), expected)
  }
})

test_that("multidimensional indexing requires one argument per axis", {
  x <- mlx_array(1:24, c(2, 3, 4))
  expect_error(x[1, ], "number of indices")
  expect_error(x[1, ] <- 0, "number of indices")
})

test_that("character indices read values and preserve names on replacement", {
  mat <- matrix(1:12, 3, 4, dimnames = list(letters[1:3], LETTERS[1:4]))
  x <- as_mlx(mat)
  expect_equal(as.matrix(x[c("c", "a"), c("D", "B")]),
               mat[c("c", "a"), c("D", "B"), drop = FALSE])
  x[c("c", "a"), c("D", "B")] <- 101:104
  mat[c("c", "a"), c("D", "B")] <- 101:104
  expect_equal(as.matrix(x), mat)
  expect_error(x["missing", ] <- 0, "not found")
  expect_error(x[, c(TRUE, NA)], "NA")
  expect_error(x[, c(TRUE, NA)] <- 0, "NA")
})

test_that("empty source axes stay empty for reads and assignments", {
  for (shape in list(c(0, 3), c(2, 0, 3))) {
    x <- as_mlx(array(numeric(0), shape))
    expect_equal(as.array(x[]), array(numeric(0), shape))
    x[] <- numeric(0)
    expect_equal(as.array(x), array(numeric(0), shape))
  }
})

test_that("slices and sparse indices preserve float64 on CPU", {
  with_device("cpu", {
    mat <- matrix(seq_len(20) / 7, 4, 5)
    x <- as_mlx(mat, dtype = "float64")
    expect_equal(as.matrix(x[2:3, 2:4]), mat[2:3, 2:4, drop = FALSE], tolerance = 1e-14)
    expect_equal(mlx_dtype(x[2:3, 2:4]), "float64")
    expect_equal(as.matrix(x[c(4L, 1L), c(5L, 2L)]),
                 mat[c(4L, 1L), c(5L, 2L), drop = FALSE], tolerance = 1e-14)
    x[2:3, 2:4] <- c(1 / 3, 1 / 7)
    mat[2:3, 2:4] <- c(1 / 3, 1 / 7)
    expect_equal(as.matrix(x), mat, tolerance = 1e-14)
    expect_equal(mlx_dtype(x), "float64")
  })
})

test_that("omitted indices can be forwarded through a caller's arguments", {
  mat <- matrix(1:12, 3, 4)
  x <- as_mlx(mat)
  # Test closures exercise missing promises, rather than literal blank expressions.
  read <- function(x, i, j) x[i, j]
  write <- function(x, i, j, value) { x[i, j] <- value; x }
  expect_equal(as.matrix(read(x, , 2L)), mat[, 2L, drop = FALSE])
  expect_equal(as.matrix(read(x, 2L, )), mat[2L, , drop = FALSE])
  mat[, 2L] <- 101:103
  expect_equal(as.matrix(write(x, , 2L, 101:103)), mat)
  mat <- matrix(1:12, 3, 4)
  mat[2L, ] <- 101:104
  expect_equal(as.matrix(write(x, 2L, , 101:104)), mat)
  expect_error(read(x, stop("missing value in custom index"), 1L),
               "missing value in custom index")
})

test_that("per-axis logical matrices behave like logical vectors", {
  mat <- matrix(1:12, 3, 4)
  mask <- matrix(c(TRUE, FALSE, TRUE), ncol = 1)
  for (convert in list(identity, as_mlx)) {
    x <- as_mlx(mat)
    expect_equal(as.matrix(x[convert(mask), ]), mat[mask, , drop = FALSE])
    x[convert(mask), ] <- 101:104
    expected <- mat
    expected[mask, ] <- 101:104
    expect_equal(as.matrix(x), expected)
  }
})

test_that("vector logical indices can have FALSE entries beyond the vector length", {
  for (convert in list(identity, as_mlx)) {
    x <- as_mlx(1:3)
    idx <- c(TRUE, FALSE, FALSE, FALSE)
    expect_equal(as.vector(x[convert(idx)]), (1:3)[idx])
    x[convert(idx)] <- 100
    expect_equal(as.vector(x), c(100, 2, 3))
    expect_error(x[convert(c(FALSE, FALSE, FALSE, TRUE))], "out of bounds")
    expect_error(x[convert(c(FALSE, FALSE, FALSE, TRUE))] <- 0, "out of bounds")
  }
})

test_that("numeric coordinate matrices cannot be used as per-axis indices", {
  x <- mlx_matrix(1:12, 3, 4)
  for (convert in list(identity, as_mlx)) {
    idx <- convert(matrix(c(1L, 3L), ncol = 1))
    expect_error(x[idx, ], "undimensioned vectors")
    expect_error(x[idx, ] <- 0, "undimensioned vectors")
  }
})
