skip_on_cran()

test_that("rbind/cbind concatenate MLX tensors", {
  x <- as_mlx(matrix(1:6, 3, 2))
  y <- as_mlx(matrix(7:12, 3, 2))

  rb <- rbind(x, y)
  cb <- cbind(x, y)

  expect_s3_class(rb, "mlx")
  expect_s3_class(cb, "mlx")
  expect_equal(rb$dim, c(6L, 2L))
  expect_equal(cb$dim, c(3L, 4L))
  expect_equal(as.matrix(rb), rbind(matrix(1:6, 3, 2), matrix(7:12, 3, 2)))
  expect_equal(as.matrix(cb), cbind(matrix(1:6, 3, 2), matrix(7:12, 3, 2)))
})

test_that("rbind/cbind coerce base matrices", {
  x <- as_mlx(matrix(1:6, 3, 2))
  rb2 <- rbind(x, matrix(13:18, 3, 2))
  cb2 <- cbind(x, matrix(13:18, 3, 2))
  expect_equal(as.matrix(rb2), rbind(matrix(1:6, 3, 2), matrix(13:18, 3, 2)))
  expect_equal(as.matrix(cb2), cbind(matrix(1:6, 3, 2), matrix(13:18, 3, 2)))
})

test_that("rbind/cbind work on 3D arrays", {
  x <- as_mlx(array(1:24, c(2, 3, 4)))
  y <- as_mlx(array(25:48, c(2, 3, 4)))

  rb <- rbind(x, y)
  cb <- cbind(x, y)

  # Check dimensions are preserved correctly
  expect_equal(dim(rb), c(4L, 3L, 4L))
  expect_equal(dim(cb), c(2L, 6L, 4L))

  # Check first slice values by dropping the third dimension
  expect_equal(drop(as.matrix(rb[, , 1])),
               rbind(matrix(1:6, 2, 3), matrix(25:30, 2, 3)))
  expect_equal(drop(as.matrix(cb[, , 1])),
               cbind(matrix(1:6, 2, 3), matrix(25:30, 2, 3)))
})

test_that("rbind/cbind work on 4D arrays", {
  x <- as_mlx(array(1:24, c(2, 3, 2, 2)))
  y <- as_mlx(array(25:48, c(2, 3, 2, 2)))

  rb <- rbind(x, y)
  cb <- cbind(x, y)

  # Check dimensions are preserved correctly
  expect_equal(dim(rb), c(4L, 3L, 2L, 2L))
  expect_equal(dim(cb), c(2L, 6L, 2L, 2L))
})
