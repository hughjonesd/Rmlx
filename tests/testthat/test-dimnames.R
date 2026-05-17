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
  expect_equal(dimnames(x[c("r1", "r3"), ]),
               list(c("r1", "r3"), colnames(mat)))
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

test_that("linalg helpers preserve base-compatible dimnames", {
  a <- matrix(c(2, 1, 1, 2), 2, 2,
              dimnames = list(c("ar1", "ar2"), c("ac1", "ac2")))
  b <- matrix(1:4, 2, 2,
              dimnames = list(c("br1", "br2"), c("bc1", "bc2")))
  x <- as_mlx(a)

  expect_equal(dimnames(solve(x, device = "cpu")), dimnames(solve(a)))
  expect_equal(
    dimnames(solve(x, as_mlx(b), device = "cpu")),
    dimnames(solve(a, b))
  )
  expect_equal(names(solve(x, c(br1 = 1, br2 = 2), device = "cpu")),
               names(solve(a, c(br1 = 1, br2 = 2))))

  spd <- crossprod(a) + diag(2)
  dimnames(spd) <- dimnames(a)
  expect_equal(dimnames(chol(as_mlx(spd), device = "cpu")), dimnames(chol(spd)))
  expect_equal(dimnames(mlx_inv(x, device = "cpu")), dimnames(solve(a)))
  expect_equal(dimnames(mlx_tri_inv(x, upper = TRUE, device = "cpu")), dimnames(solve(a)))
  expect_equal(dimnames(pinv(x, device = "cpu")), list(colnames(a), rownames(a)))

  qr_mlx <- qr(x, device = "cpu")
  expect_equal(dimnames(qr_mlx$Q), list(rownames(a), colnames(a)))
  expect_equal(dimnames(qr_mlx$R), list(colnames(a), colnames(a)))

  xv <- c(x1 = 1, x2 = 2)
  yv <- c(y1 = 3, y2 = 4, y3 = 5)
  expect_equal(dimnames(outer(as_mlx(xv), as_mlx(yv))), dimnames(outer(xv, yv)))
})

test_that("diagonal helpers match base-compatible names", {
  named <- matrix(1:9, 3, 3, dimnames = list(c("a", "b", "c"), c("a", "b", "c")))
  unmatched <- matrix(1:6, 2, 3,
                      dimnames = list(c("r1", "r2"), c("c1", "c2", "c3")))

  expect_equal(names(diag(as_mlx(named))), names(diag(named)))
  expect_null(names(diag(as_mlx(named), names = FALSE)))
  expect_equal(names(diag(as_mlx(unmatched))), names(diag(unmatched)))
  expect_null(dimnames(diag(as_mlx(c(a = 1, b = 2)))))

  arr <- array(1:12, c(2, 2, 3),
               dimnames = list(c("i1", "i2"), c("i1", "i2"), c("b1", "b2", "b3")))
  diagonal <- mlx_diagonal(as_mlx(arr), axis1 = 1, axis2 = 2)
  expect_equal(dimnames(diagonal), list(c("b1", "b2", "b3"), c("i1", "i2")))
})

test_that("triangular solves use solve-style names where base does", {
  a <- matrix(c(2, 1, 0, 3), 2, 2,
              dimnames = list(c("ar1", "ar2"), c("ac1", "ac2")))
  b <- matrix(c(1, 5, 2, 6), 2, 2,
              dimnames = list(c("br1", "br2"), c("bc1", "bc2")))

  triangular <- mlx_solve_triangular(as_mlx(a), as_mlx(b), upper = FALSE, device = "cpu")
  expect_equal(dimnames(triangular), dimnames(solve(a, b)))

  triangular_vec <- mlx_solve_triangular(
    as_mlx(a),
    as_mlx(c(br1 = 1, br2 = 5)),
    upper = FALSE,
    device = "cpu"
  )
  expect_equal(names(triangular_vec), names(solve(a, c(br1 = 1, br2 = 5))))

  backsolve_res <- backsolve(as_mlx(a), as_mlx(b), upper.tri = FALSE, device = "cpu")
  expect_equal(dimnames(backsolve_res), dimnames(base::backsolve(a, b, upper.tri = FALSE)))
})

test_that("transform helpers preserve dimnames when positions still correspond", {
  mat <- matrix(c(1, 2, 3, 4, 5, 6), 2, 3, byrow = TRUE,
                dimnames = list(c("r1", "r2"), c("c1", "c2", "c3")))
  x <- as_mlx(mat)

  expect_equal(dimnames(mlx_softmax(x, axes = 2)), dimnames(mat))
  expect_equal(dimnames(mlx_logcumsumexp(x, axis = 2)), dimnames(mat))
  expect_equal(dimnames(mlx_fft(x)), dimnames(fft(mat)))
  expect_equal(names(mlx_fft(as_mlx(c(a = 1, b = 2, c = 3)))),
               names(fft(c(a = 1, b = 2, c = 3))))
  expect_equal(dimnames(mlx_hadamard_transform(x[, 1:2])), dimnames(x[, 1:2]))

  expect_equal(names(mlx_argmax(x, axis = 2)), rownames(mat))
  expect_equal(names(mlx_argmin(x, axis = 1)), colnames(mat))
  expect_equal(names(mlx_norm(x, axes = 2)), rownames(mat))
  expect_equal(dimnames(mlx_logsumexp(x, axes = 2, drop = FALSE)),
               list(rownames(mat), NULL))
  expect_null(dimnames(mlx_logsumexp(x)))
})

test_that("shape-preserving helpers keep dimnames", {
  mat <- matrix(1:12, 3, 4,
                dimnames = list(c("r1", "r2", "r3"), c("c1", "c2", "c3", "c4")))
  x <- as_mlx(mat)

  expect_equal(dimnames(mlx_clip(x, min = 3, max = 10)), dimnames(mat))
  expect_equal(dimnames(mlx_zeros_like(x)), dimnames(mat))
  expect_equal(dimnames(mlx_ones_like(x)), dimnames(mat))
  expect_equal(dimnames(mlx_tril(x)), dimnames(mat))
  expect_equal(dimnames(mlx_triu(x)), dimnames(mat))
  expect_equal(dimnames(mlx_isclose(x, x + 1)), dimnames(mat))
  expect_equal(dimnames(mlx_cumsum(x, axis = 1)), dimnames(mat))
  expect_equal(dimnames(mlx_cumprod(x, axis = 2)), dimnames(mat))

  named <- as_mlx(c(a = 1, b = 2, c = 3))
  expect_equal(names(cumsum(named)), c("a", "b", "c"))
  expect_equal(names(cumprod(named)), c("a", "b", "c"))
  expect_equal(names(cummax(named)), c("a", "b", "c"))
  expect_equal(names(cummin(named)), c("a", "b", "c"))

  updated <- mlx_slice_update(
    x,
    mlx_matrix(100:103, nrow = 2),
    start = c(1L, 2L),
    stop = c(2L, 3L)
  )
  expect_equal(dimnames(updated), dimnames(mat))

  idx <- matrix(c(1L, 4L,
                  2L, 3L,
                  4L, 1L), nrow = 3, byrow = TRUE)
  values <- matrix(c(100, 200, 300, 400, 500, 600), nrow = 3, byrow = TRUE)
  expect_equal(dimnames(mlx_put_along_axis(x, idx, values, axis = 2L)), dimnames(mat))

  scatter_idx <- matrix(c(1L, 1L,
                          2L, 3L,
                          4L, 4L), nrow = 3, byrow = TRUE)
  expect_equal(dimnames(mlx_scatter_add_axis(x, scatter_idx, values, axis = 2L)),
               dimnames(mat))
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

  stacked <- mlx_stack(x, x, axis = 1)
  expect_equal(dimnames(stacked), list(NULL, rownames(x), colnames(x)))

  expanded <- mlx_expand_dims(x, axes = 2)
  expect_equal(dimnames(expanded), list(rownames(x), NULL, colnames(x)))

  expect_equal(dimnames(mlx_where(x > 2, x, x)), dimnames(x))
  expect_equal(dimnames(mlx_where(x > 2, 1, 0)), dimnames(x))
})

test_that("axis movement and splitting transform dimnames like base analogues", {
  arr <- array(1:24, c(2, 3, 4),
               dimnames = list(c("a1", "a2"),
                               c("b1", "b2", "b3"),
                               c("c1", "c2", "c3", "c4")))
  x <- as_mlx(arr)

  expect_equal(dimnames(mlx_moveaxis(x, source = 1, destination = 3)),
               dimnames(aperm(arr, c(2, 3, 1))))
  expect_equal(dimnames(mlx_moveaxis(x, source = c(1, 3), destination = c(3, 1))),
               dimnames(aperm(arr, c(3, 2, 1))))

  parts <- mlx_split(x, sections = 3, axis = 2)
  expect_equal(dimnames(parts[[2]]), list(dimnames(arr)[[1]], "b2", dimnames(arr)[[3]]))

  custom <- mlx_split(x, sections = list(1), axis = 2)
  expect_equal(dimnames(custom[[2]]), list(dimnames(arr)[[1]], c("b2", "b3"),
                                           dimnames(arr)[[3]]))

  base_split <- base::asplit(arr, 2)
  mlx_split <- asplit(x, 2)
  expect_equal(names(mlx_split), names(base_split))
  expect_equal(dimnames(mlx_split[[1]]), dimnames(base_split[[1]]))
})
