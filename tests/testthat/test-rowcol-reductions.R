test_that("row/col sums follow dims semantics for mlx arrays", {
  x <- array(1:24, dim = c(2, 3, 4))
  mx <- as_mlx(x)

  expect_equal(
    as_r(rowSums(mx, dims = 1)),
    rowSums(x, dims = 1),
    tolerance = 1e-6
  )
  expect_equal(
    as_r(rowSums(mx, dims = 2)),
    rowSums(x, dims = 2),
    tolerance = 1e-6
  )
  expect_equal(
    as_r(colSums(mx, dims = 1)),
    colSums(x, dims = 1),
    tolerance = 1e-6
  )
  expect_equal(
    as_r(colSums(mx, dims = 2)),
    colSums(x, dims = 2),
    tolerance = 1e-6
  )
})

test_that("row/col means follow dims semantics for mlx arrays", {
  x <- array(1:24, dim = c(2, 3, 4))
  mx <- as_mlx(x)

  expect_equal(
    as_r(rowMeans(mx, dims = 1)),
    rowMeans(x, dims = 1),
    tolerance = 1e-6
  )
  expect_equal(
    as_r(rowMeans(mx, dims = 2)),
    rowMeans(x, dims = 2),
    tolerance = 1e-6
  )
  expect_equal(
    as_r(colMeans(mx, dims = 1)),
    colMeans(x, dims = 1),
    tolerance = 1e-6
  )
  expect_equal(
    as_r(colMeans(mx, dims = 2)),
    colMeans(x, dims = 2),
    tolerance = 1e-6
  )
})
