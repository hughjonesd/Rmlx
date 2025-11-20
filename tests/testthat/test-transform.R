hadamard_matrix <- function(n) {
  if (n == 1L) {
    return(matrix(1, 1, 1))
  }
  if (n %% 2L != 0L) {
    stop("Hadamard matrices are defined for powers of two.", call. = FALSE)
  }
  H <- hadamard_matrix(n / 2L)
  rbind(cbind(H, H), cbind(H, -H))
}

test_that("mlx_hadamard_transform matches reference implementation", {
  x <- as_mlx(c(1, -1))
  res <- as.vector(mlx_hadamard_transform(x))
  expected <- as.vector((hadamard_matrix(2) / sqrt(2)) %*% c(1, -1))
  expect_equal(res, expected, tolerance = 1e-6)

  res_raw <- as.vector(mlx_hadamard_transform(x, scale = 1))
  expected_raw <- as.vector(hadamard_matrix(2) %*% c(1, -1))
  expect_equal(res_raw, expected_raw, tolerance = 1e-6)

  mat <- matrix(1:8, nrow = 2, byrow = TRUE)
  mlx_mat <- as_mlx(mat)
  res_mat <- as.matrix(mlx_hadamard_transform(mlx_mat))
  H4 <- hadamard_matrix(4) / sqrt(4)
  expected_mat <- t(apply(mat, 1, function(row) H4 %*% row))
  expect_equal(res_mat, expected_mat, tolerance = 1e-6)

  expect_error(as.matrix(mlx_hadamard_transform(as_mlx(c(1, 0, -1)))))
})

test_that("mlx_hadamard_transform handles higher dimensional arrays", {
  arr <- array(seq_len(16), dim = c(2, 2, 4))
  res <- as.array(mlx_hadamard_transform(arr))

  H4 <- hadamard_matrix(4) / sqrt(4)
  expected_tmp <- apply(arr, c(1, 2), function(slice) {
    as.vector(H4 %*% slice)
  })
  expected <- aperm(expected_tmp, c(2, 3, 1))

  expect_equal(res, expected, tolerance = 1e-6)
})

test_that("mlx_argmax and mlx_argmin match base behaviour", {
  x <- c(-3, 5, 1, 9, 9, -4)
  t <- as_mlx(x)

  argmax <- as.vector(mlx_argmax(t))
  argmin <- as.vector(mlx_argmin(t))

  expect_equal(as.integer(argmax), which.max(x))
  expect_equal(as.integer(argmin), which.min(x))

  mat <- matrix(c(1, 7, 3,
                  9, 2, 4), nrow = 2, byrow = TRUE)
  m_t <- as_mlx(mat)

  col_argmax <- mlx_argmax(m_t, axis = 1L)
  row_argmax <- mlx_argmax(m_t, axis = 2L)

  expect_equal(as.integer(col_argmax), apply(mat, 2, which.max))
  expect_equal(as.integer(row_argmax), apply(mat, 1, which.max))
})

test_that("mlx_sort and mlx_argsort agree with base R", {
  x <- c(3, -1, 5, 2)
  t <- as_mlx(x)

  sorted_vals <- as.vector(mlx_sort(t))
  expect_equal(sorted_vals, sort(x))

  idx <- as.integer(as.vector(mlx_argsort(t)))
  # argsort now returns 1-based indices
  expect_equal(x[idx], sort(x))

  mat <- matrix(c(3, 4, 1,
                  9, 2, 5), nrow = 2, byrow = TRUE)
  m_t <- as_mlx(mat)
  sorted_axis2 <- as.matrix(mlx_sort(m_t, axis = 2L))
  expected <- t(apply(mat, 1, sort))
  expect_equal(sorted_axis2, expected)
})

test_that("mlx_topk returns the expected values", {
  x <- as_mlx(c(0.4, 2, -1, 3, 7))
  top2 <- sort(as.vector(mlx_topk(x, 2L)))
  expect_equal(top2, tail(sort(c(0.4, 2, -1, 3, 7)), 2))

  mat <- matrix(c(1, 5, 2,
                  8, 3, 4), nrow = 2, byrow = TRUE)
  m_t <- as_mlx(mat)
  top_axis <- mlx_topk(m_t, 2L, axis = 2L)
  observed_sorted <- t(apply(as.matrix(top_axis), 1, sort))
  expected <- t(apply(mat, 1, function(row) tail(sort(row), 2)))
  expect_equal(observed_sorted, expected)
})

test_that("mlx_partition and mlx_argpartition position kth elements", {
  x <- as_mlx(c(5, 1, 7, 3, 9))
  kth <- 2L

  part <- as.vector(mlx_partition(x, kth))
  ref_sorted <- sort(c(5, 1, 7, 3, 9))
  expect_equal(part[kth + 1L], ref_sorted[kth + 1L])
  expect_true(all(part[seq_len(kth + 1L)] <= part[kth + 1L]))

  argpart <- as.integer(as.vector(mlx_argpartition(x, kth)))
  # argpart now returns 1-based indices
  expect_equal(c(5, 1, 7, 3, 9)[argpart][kth + 1L], ref_sorted[kth + 1L])
})

test_that("mlx_logsumexp matches base computations", {
  vec <- c(-2, -1, 0, 1)
  mlx_vec <- as_mlx(vec)
  expect_equal(
    as.numeric(mlx_logsumexp(mlx_vec)),
    log(sum(exp(vec))),
    tolerance = 1e-6
  )

  mat <- matrix(seq_len(6), nrow = 2)
  mlx_mat <- as_mlx(mat)
  lse_axis2 <- mlx_logsumexp(mlx_mat, axes = 2)
  expected <- apply(mat, 1, function(row) log(sum(exp(row))))
  expect_equal(as.numeric(lse_axis2), expected, tolerance = 1e-6)
})

test_that("mlx_logcumsumexp matches cumulative reference", {
  vec <- c(-1, 0, 2)
  mlx_vec <- as_mlx(vec)
  expect_equal(
    as.vector(mlx_logcumsumexp(mlx_vec)),
    log(cumsum(exp(vec))),
    tolerance = 1e-6
  )

  mat <- matrix(c(0, 1, 2, 3, 4, 5), nrow = 2, byrow = TRUE)
  mlx_mat <- as_mlx(mat)
  res <- as.matrix(mlx_logcumsumexp(mlx_mat, axis = 2))
  expected <- t(apply(mat, 1, function(row) log(cumsum(exp(row)))))
  expect_equal(res, expected, tolerance = 1e-6)
})

test_that("mlx_softmax normalizes along the requested axis", {
  mat <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, byrow = TRUE)
  mlx_mat <- as_mlx(mat)
  sm_rows <- as.matrix(mlx_softmax(mlx_mat, axes = 2))
  expected_rows <- t(apply(mat, 1, function(row) {
    ex <- exp(row - max(row))
    ex / sum(ex)
  }))
  expect_equal(sm_rows, expected_rows, tolerance = 1e-6)

  sm_cols <- as.matrix(mlx_softmax(mlx_mat, axes = 1))
  expected_cols <- apply(mat, 2, function(col) {
    ex <- exp(col - max(col))
    ex / sum(ex)
  })
  expect_equal(sm_cols, expected_cols, tolerance = 1e-6)
})

test_that("mlx_argsort works with axis argument on 2D arrays", {
  mat <- matrix(c(3, 1, 4,
                  2, 5, 0), nrow = 2, byrow = TRUE)
  m_mlx <- as_mlx(mat)

  # Sort along axis 1 (columns)
  idx_axis1 <- as.matrix(mlx_argsort(m_mlx, axis = 1))
  # For each column, indices should give sorted order
  for (j in seq_len(ncol(mat))) {
    col_values <- mat[, j]
    sorted_values <- col_values[idx_axis1[, j]]
    expect_equal(sorted_values, sort(col_values))
  }

  # Sort along axis 2 (rows)
  idx_axis2 <- as.matrix(mlx_argsort(m_mlx, axis = 2))
  # For each row, indices should give sorted order
  for (i in seq_len(nrow(mat))) {
    row_values <- mat[i, ]
    sorted_values <- row_values[idx_axis2[i, ]]
    expect_equal(sorted_values, sort(row_values))
  }
})

test_that("mlx_argsort works with axis argument on 3D arrays", {
  arr <- array(c(3, 1, 4, 2,
                 5, 0, 6, 7), dim = c(2, 2, 2))
  arr_mlx <- as_mlx(arr)

  # Sort along axis 1
  idx_axis1 <- as.array(mlx_argsort(arr_mlx, axis = 1))
  expect_equal(dim(idx_axis1), dim(arr))

  # Check a specific slice - sort along axis 1 at position [,1,1]
  expect_equal(arr[idx_axis1[, 1, 1], 1, 1], sort(arr[, 1, 1]))

  # Sort along axis 3
  idx_axis3 <- as.array(mlx_argsort(arr_mlx, axis = 3))
  # Check sorting along the last axis
  expect_equal(arr[1, 1, idx_axis3[1, 1, ]], sort(arr[1, 1, ]))
})

test_that("mlx_argmax works with axis argument on 2D arrays", {
  mat <- matrix(c(3, 1, 4,
                  9, 5, 2), nrow = 2, byrow = TRUE)
  m_mlx <- as_mlx(mat)

  # argmax along axis 1 (columns)
  argmax_axis1 <- as.vector(mlx_argmax(m_mlx, axis = 1))
  expected_axis1 <- apply(mat, 2, which.max)
  expect_equal(as.integer(argmax_axis1), expected_axis1)

  # argmax along axis 2 (rows)
  argmax_axis2 <- as.vector(mlx_argmax(m_mlx, axis = 2))
  expected_axis2 <- apply(mat, 1, which.max)
  expect_equal(as.integer(argmax_axis2), expected_axis2)
})

test_that("mlx_argmin works with axis argument on 2D arrays", {
  mat <- matrix(c(3, 1, 4,
                  9, 5, 2), nrow = 2, byrow = TRUE)
  m_mlx <- as_mlx(mat)

  # argmin along axis 1 (columns)
  argmin_axis1 <- as.vector(mlx_argmin(m_mlx, axis = 1))
  expected_axis1 <- apply(mat, 2, which.min)
  expect_equal(as.integer(argmin_axis1), expected_axis1)

  # argmin along axis 2 (rows)
  argmin_axis2 <- as.vector(mlx_argmin(m_mlx, axis = 2))
  expected_axis2 <- apply(mat, 1, which.min)
  expect_equal(as.integer(argmin_axis2), expected_axis2)
})

test_that("mlx_argmax and mlx_argmin work with 3D arrays", {
  arr <- array(c(1, 5, 3, 2,
                 9, 0, 7, 4), dim = c(2, 2, 2))
  arr_mlx <- as_mlx(arr)

  # argmax along axis 1
  argmax_axis1 <- as.array(mlx_argmax(arr_mlx, axis = 1))
  expect_equal(dim(argmax_axis1), c(2, 2))
  expect_equal(argmax_axis1[1, 1], which.max(arr[, 1, 1]))

  # argmin along axis 3
  argmin_axis3 <- as.array(mlx_argmin(arr_mlx, axis = 3))
  expect_equal(dim(argmin_axis3), c(2, 2))
  expect_equal(argmin_axis3[1, 1], which.min(arr[1, 1, ]))
})

test_that("mlx_logsumexp works with axis argument on 2D arrays", {
  mat <- matrix(c(1, 2, 3,
                  4, 5, 6), nrow = 2, byrow = TRUE)
  m_mlx <- as_mlx(mat)

  # logsumexp along axis 1 (columns)
  lse_axis1 <- as.vector(mlx_logsumexp(m_mlx, axes = 1))
  expected_axis1 <- apply(mat, 2, function(col) log(sum(exp(col))))
  expect_equal(lse_axis1, expected_axis1, tolerance = 1e-6)

  # logsumexp along axis 2 (rows)
  lse_axis2 <- as.vector(mlx_logsumexp(m_mlx, axes = 2))
  expected_axis2 <- apply(mat, 1, function(row) log(sum(exp(row))))
  expect_equal(lse_axis2, expected_axis2, tolerance = 1e-6)
})

test_that("mlx_logsumexp works with 3D arrays", {
  arr <- array(1:8, dim = c(2, 2, 2))
  arr_mlx <- as_mlx(arr)

  # logsumexp along axis 3
  lse_axis3 <- as.array(mlx_logsumexp(arr_mlx, axes = 3))
  expect_equal(dim(lse_axis3), c(2, 2))
  expected <- log(sum(exp(arr[1, 1, ])))
  expect_equal(lse_axis3[1, 1], expected, tolerance = 1e-6)
})

test_that("mlx_softmax works with different axes", {
  mat <- matrix(c(1, 2, 3,
                  4, 5, 6), nrow = 2, byrow = TRUE)
  m_mlx <- as_mlx(mat)

  # softmax along axis 1 (columns)
  sm_axis1 <- as.matrix(mlx_softmax(m_mlx, axes = 1))
  expected_axis1 <- apply(mat, 2, function(col) {
    ex <- exp(col - max(col))
    ex / sum(ex)
  })
  expect_equal(sm_axis1, expected_axis1, tolerance = 1e-6)

  # Verify columns sum to 1
  expect_equal(colSums(sm_axis1), rep(1, ncol(sm_axis1)), tolerance = 1e-6)

  # softmax along axis 2 (rows)
  sm_axis2 <- as.matrix(mlx_softmax(m_mlx, axes = 2))
  expected_axis2 <- t(apply(mat, 1, function(row) {
    ex <- exp(row - max(row))
    ex / sum(ex)
  }))
  expect_equal(sm_axis2, expected_axis2, tolerance = 1e-6)

  # Verify rows sum to 1
  expect_equal(rowSums(sm_axis2), rep(1, nrow(sm_axis2)), tolerance = 1e-6)
})

test_that("mlx_partition works with axis argument on 2D arrays", {
  mat <- matrix(c(5, 1, 7, 3,
                  9, 2, 6, 4), nrow = 2, byrow = TRUE)
  m_mlx <- as_mlx(mat)
  kth <- 1L  # Second element (0-indexed)

  # Partition along axis 2 (rows)
  part_axis2 <- as.matrix(mlx_partition(m_mlx, kth, axis = 2))
  # For each row, kth element should be in correct position
  for (i in seq_len(nrow(mat))) {
    row_part <- part_axis2[i, ]
    row_sorted <- sort(mat[i, ])
    # Element at position kth+1 should match sorted
    expect_equal(row_part[kth + 1], row_sorted[kth + 1])
    # Elements before kth should be <= kth element
    expect_true(all(row_part[seq_len(kth + 1)] <= row_part[kth + 1]))
  }

  # Partition along axis 1 (columns)
  part_axis1 <- as.matrix(mlx_partition(m_mlx, kth, axis = 1))
  for (j in seq_len(ncol(mat))) {
    col_part <- part_axis1[, j]
    col_sorted <- sort(mat[, j])
    expect_equal(col_part[kth + 1], col_sorted[kth + 1])
    expect_true(all(col_part[seq_len(kth + 1)] <= col_part[kth + 1]))
  }
})

test_that("mlx_partition works with axis argument on 3D arrays", {
  arr <- array(c(8, 2, 5, 1,
                 9, 3, 7, 4), dim = c(2, 2, 2))
  arr_mlx <- as_mlx(arr)
  kth <- 0L  # First element (0-indexed)

  # Partition along axis 3
  part_axis3 <- as.array(mlx_partition(arr_mlx, kth, axis = 3))
  expect_equal(dim(part_axis3), dim(arr))

  # Check specific slice [1,1,]
  slice_part <- part_axis3[1, 1, ]
  slice_sorted <- sort(arr[1, 1, ])
  expect_equal(slice_part[kth + 1], slice_sorted[kth + 1])
  expect_true(all(slice_part[seq_len(kth + 1)] <= slice_part[kth + 1]))
})


test_that("mlx_argpartition works with axis argument on 2D arrays", {
  mat <- matrix(c(5, 1, 7, 3,
                  9, 2, 6, 4), nrow = 2, byrow = TRUE)
  m_mlx <- as_mlx(mat)
  kth <- 1L

  # Argpartition along axis 2 (rows)
  idx_axis2 <- as.matrix(mlx_argpartition(m_mlx, kth, axis = 2))
  for (i in seq_len(nrow(mat))) {
    row_values <- mat[i, ]
    row_partitioned <- row_values[idx_axis2[i, ]]
    row_sorted <- sort(row_values)
    # Element at kth position should match
    expect_equal(row_partitioned[kth + 1], row_sorted[kth + 1])
    # Elements before should be smaller or equal
    expect_true(all(row_partitioned[seq_len(kth + 1)] <= row_partitioned[kth + 1]))
  }

  # Argpartition along axis 1 (columns)
  idx_axis1 <- as.matrix(mlx_argpartition(m_mlx, kth, axis = 1))
  for (j in seq_len(ncol(mat))) {
    col_values <- mat[, j]
    col_partitioned <- col_values[idx_axis1[, j]]
    col_sorted <- sort(col_values)
    expect_equal(col_partitioned[kth + 1], col_sorted[kth + 1])
    expect_true(all(col_partitioned[seq_len(kth + 1)] <= col_partitioned[kth + 1]))
  }
})

test_that("mlx_argpartition works with axis argument on 3D arrays", {
  arr <- array(c(8, 2, 5, 1,
                 9, 3, 7, 4), dim = c(2, 2, 2))
  arr_mlx <- as_mlx(arr)
  kth <- 0L

  # Argpartition along axis 3
  idx_axis3 <- as.array(mlx_argpartition(arr_mlx, kth, axis = 3))
  expect_equal(dim(idx_axis3), dim(arr))

  # Check specific slice
  slice_indices <- idx_axis3[1, 1, ]
  slice_values <- arr[1, 1, ]
  slice_partitioned <- slice_values[slice_indices]
  slice_sorted <- sort(slice_values)

  expect_equal(slice_partitioned[kth + 1], slice_sorted[kth + 1])
  expect_true(all(slice_partitioned[seq_len(kth + 1)] <= slice_partitioned[kth + 1]))
})

