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

  col_argmax <- as.matrix(mlx_argmax(m_t, axis = 1L))
  row_argmax <- as.matrix(mlx_argmax(m_t, axis = 2L))

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
    as.numeric(as.matrix(mlx_logsumexp(mlx_vec))),
    log(sum(exp(vec))),
    tolerance = 1e-6
  )

  mat <- matrix(seq_len(6), nrow = 2)
  mlx_mat <- as_mlx(mat)
  lse_axis2 <- as.matrix(mlx_logsumexp(mlx_mat, axes = 2))
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
