test_that("constructors and conversion preserve dimnames", {
  mat <- matrix(1:6, 2, 3, dimnames = list(c("r1", "r2"), c("c1", "c2", "c3")))
  x <- as_mlx(mat)

  expect_equal(dimnames(x), dimnames(mat))
  expect_equal(dimnames(as.array(x)), dimnames(mat))
  expect_equal(dimnames(as.matrix(x)), dimnames(mat))

  vec <- c(a = 1, b = 2, c = 3)
  xv <- as_mlx(vec)
  expect_equal(names(xv), names(vec))
  expect_equal(names(as_r(xv)), names(vec))

  y <- mlx_matrix(1:4, 2, 2, dimnames = list(c("ra", "rb"), c("ca", "cb")))
  expect_equal(rownames(y), c("ra", "rb"))
  expect_equal(colnames(y), c("ca", "cb"))

  z <- mlx_vector(setNames(1:3, c("x", "y", "z")))
  expect_equal(names(z), c("x", "y", "z"))
})

test_that("dimname setters validate shape", {
  x <- mlx_matrix(1:4, 2, 2)
  dimnames(x) <- list(c("a", "b"), c("c", "d"))
  expect_equal(dimnames(x), list(c("a", "b"), c("c", "d")))

  rownames(x) <- c("r1", "r2")
  colnames(x) <- c("c1", "c2")
  expect_equal(rownames(x), c("r1", "r2"))
  expect_equal(colnames(x), c("c1", "c2"))

  expect_error(dimnames(x) <- list("too-short", c("c1", "c2")), "must match dimension")
  expect_error(names(x) <- c("a", "b"), "one-dimensional")
})

test_that("subsetting updates dimnames and supports character indices", {
  mat <- matrix(1:9, 3, 3, dimnames = list(c("r1", "r2", "r3"), c("c1", "c2", "c3")))
  x <- as_mlx(mat)

  expect_equal(dimnames(x[c("r3", "r1"), c("c2", "c2")]),
               list(c("r3", "r1"), c("c2", "c2")))
  expect_equal(names(x["r2", , drop = TRUE]), colnames(mat))
  expect_equal(names(x[, "c3", drop = TRUE]), rownames(mat))
  expect_equal(dimnames(x[matrix(c(1, 1, 3, 2), ncol = 2, byrow = TRUE)]), NULL)

  expect_error(x["missing", ], "not found")
  expect_error(mlx_matrix(1:4, 2, 2)["a", ], "require dimnames")
})

test_that("subset replacement preserves target dimnames", {
  x <- mlx_matrix(1:4, 2, 2, dimnames = list(c("r1", "r2"), c("c1", "c2")))
  x["r2", "c1"] <- mlx_matrix(99, 1, 1, dimnames = list("other-row", "other-col"))

  expect_equal(dimnames(x), list(c("r1", "r2"), c("c1", "c2")))
  expect_equal(as.matrix(x)["r2", "c1"], 99)
})

test_that("common operations preserve or transform dimnames", {
  x <- mlx_matrix(1:6, 2, 3, dimnames = list(c("r1", "r2"), c("c1", "c2", "c3")))
  y <- mlx_matrix(1:6, 2, 3, dimnames = list(c("yr1", "yr2"), c("yc1", "yc2", "yc3")))
  z <- mlx_matrix(1:6, 3, 2, dimnames = list(c("k1", "k2", "k3"), c("out1", "out2")))

  expect_equal(dimnames(-x), dimnames(x))
  expect_equal(dimnames(x + 1), dimnames(x))
  expect_equal(dimnames(1 + x), dimnames(x))
  expect_equal(dimnames(x + y), dimnames(x))
  expect_equal(dimnames(t(x)), rev(dimnames(x)))
  expect_equal(dimnames(aperm(x, c(2, 1))), rev(dimnames(x)))
  expect_equal(dimnames(x %*% z), list(rownames(x), colnames(z)))

  expect_equal(names(rowSums(x)), rownames(x))
  expect_equal(names(colMeans(x)), colnames(x))
  expect_equal(dimnames(mlx_sum(x, axes = 2, drop = FALSE)), list(rownames(x), NULL))
  expect_null(dimnames(sum(x)))
})

test_that("binding combines names on bound axis and keeps agreeing axes", {
  x <- mlx_matrix(1:4, 2, 2, dimnames = list(c("r1", "r2"), c("c1", "c2")))
  y <- mlx_matrix(5:8, 2, 2, dimnames = list(c("r3", "r4"), c("c1", "c2")))

  rb <- rbind(x, y)
  expect_equal(dimnames(rb), list(c("r1", "r2", "r3", "r4"), c("c1", "c2")))

  y_same_rows <- mlx_matrix(5:8, 2, 2, dimnames = list(c("r1", "r2"), c("c1", "c2")))
  cb <- cbind(x, y_same_rows)
  expect_equal(dimnames(cb), list(c("r1", "r2"), c("c1", "c2", "c1", "c2")))
})

test_that("shape-changing transforms drop dimnames", {
  x <- mlx_matrix(1:4, 2, 2, dimnames = list(c("r1", "r2"), c("c1", "c2")))

  expect_null(dimnames(mlx_reshape(x, 4)))
  expect_null(dimnames(mlx_flatten(x)))
})
