test_that("sum works", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  s <- as.matrix(sum(x_mlx))
  expect_equal(as.numeric(s), sum(x), tolerance = 1e-6)
})

test_that("mean works", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  m <- as.matrix(mean(x_mlx))
  expect_equal(as.numeric(m), mean(x), tolerance = 1e-6)
})

test_that("colMeans works", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  cm <- as.matrix(colMeans(x_mlx))
  expect_equal(as.numeric(cm), colMeans(x), tolerance = 1e-6)
})

test_that("rowMeans works", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  rm <- as.matrix(rowMeans(x_mlx))
  expect_equal(as.numeric(rm), rowMeans(x), tolerance = 1e-6)
})
