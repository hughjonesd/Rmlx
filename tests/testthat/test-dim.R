test_that("dim and nrow for 1D mlx vectors follow base semantics", {
  x <- as_mlx(1:5)

  expect_null(dim(x))
  expect_equal(mlx_shape(x), 5L)
  expect_null(nrow(x))
  expect_null(ncol(x))
  expect_equal(length(x), 5L)
})

test_that("dim and nrow for scalar mlx values are NULL", {
  s <- as_mlx(42)

  expect_null(dim(s))
  expect_equal(mlx_shape(s), integer(0))
  expect_null(nrow(s))
  expect_null(ncol(s))
  expect_equal(length(s), 1L)
})

test_that("mlx_shape returns dimensions for 2D arrays", {
  m <- mlx_matrix(1:6, nrow = 2)
  expect_equal(mlx_shape(m), c(2L, 3L))
  expect_equal(dim(m), c(2L, 3L))
})
