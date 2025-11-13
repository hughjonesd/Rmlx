skip_on_cran()

test_that("rbind/cbind concatenate MLX tensors", {
  x <- as_mlx(matrix(1:6, 3, 2))
  y <- as_mlx(matrix(7:12, 3, 2))

  rb <- rbind(x, y)
  cb <- cbind(x, y)

  expect_s3_class(rb, "mlx")
  expect_s3_class(cb, "mlx")
  expect_equal(dim(rb), c(6L, 2L))
  expect_equal(dim(cb), c(3L, 4L))
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

test_that("abind concatenates along arbitrary axes", {
  arr1 <- array(1:24, c(2, 3, 4))
  arr2 <- array(25:48, c(2, 3, 4))
  x <- as_mlx(arr1)
  y <- as_mlx(arr2)

  along3 <- abind(x, y, along = 3)
  expect_equal(dim(along3), c(2L, 3L, 8L))
  expected3 <- array(0, dim = c(2, 3, 8))
  expected3[, , 1:4] <- arr1
  expected3[, , 5:8] <- arr2
  expect_equal(as.matrix(along3), expected3)

  along2 <- abind(list(x, y), along = 2)
  expect_equal(dim(along2), c(2L, 6L, 4L))
  expected2 <- array(0, dim = c(2, 6, 4))
  expected2[, 1:3, ] <- arr1
  expected2[, 4:6, ] <- arr2
  expect_equal(as.matrix(along2), expected2)
})
