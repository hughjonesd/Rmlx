positive_base_qr <- function(x, y = NULL) {
  q <- qr(x)
  r <- qr.R(q)
  signs <- sign(diag(r))
  signs[signs == 0] <- 1

  out <- list(R = sweep(r, 1L, signs, `*`))
  if (!is.null(y)) {
    qty <- qr.qty(q, y)
    if (is.null(dim(qty))) {
      qty <- matrix(qty[seq_len(ncol(x))], ncol = 1L)
    } else {
      qty <- qty[seq_len(ncol(x)), , drop = FALSE]
    }
    out$qty <- sweep(qty, 1L, signs, `*`)
  }
  out
}

test_that("mlx_qr_gpu default path matches base qr quantities", {
  skip_if_not(mlx_has_gpu())

  set.seed(20260613)
  x <- matrix(rnorm(240), 60, 4)
  y <- matrix(rnorm(60), 60, 1)

  fit <- mlx_qr_gpu(as_mlx(x), as_mlx(y), block_rows = 16L)
  expected <- positive_base_qr(x, y)

  expect_equal(as.matrix(fit$R), expected$R, tolerance = 1e-4)
  expect_equal(as.matrix(fit$qty), expected$qty, tolerance = 1e-4)
  expect_equal(fit$rank, ncol(x))
  expect_equal(fit$pivot, seq_len(ncol(x)))
  expect_equal(fit$method, "cholqr")
})

test_that("mlx_qr_gpu handles matrix responses", {
  skip_if_not(mlx_has_gpu())

  set.seed(20260613)
  x <- matrix(rnorm(600), 100, 6)
  y <- matrix(rnorm(300), 100, 3)

  fit <- mlx_qr_gpu(as_mlx(x), as_mlx(y), block_rows = 20L)
  expected <- positive_base_qr(x, y)

  expect_equal(as.matrix(fit$R), expected$R, tolerance = 1e-4)
  expect_equal(as.matrix(fit$qty), expected$qty, tolerance = 1e-4)
})

test_that("mlx_qr_gpu Householder path matches base qr quantities", {
  skip_if_not(mlx_has_gpu())

  set.seed(20260613)
  x <- matrix(rnorm(240), 60, 4)
  y <- matrix(rnorm(120), 60, 2)

  fit <- mlx_qr_gpu(as_mlx(x), as_mlx(y), method = "householder")
  expected <- positive_base_qr(x, y)

  expect_equal(as.matrix(fit$R), expected$R, tolerance = 1e-4)
  expect_equal(as.matrix(fit$qty), expected$qty, tolerance = 1e-4)
  expect_equal(crossprod(as.matrix(fit$R)), crossprod(x), tolerance = 1e-4)
  expect_equal(fit$method, "householder")
})

test_that("mlx_qr_gpu custom Metal Householder path matches base qr quantities", {
  skip_if_not(mlx_has_gpu())

  set.seed(20260613)
  x <- matrix(rnorm(240), 60, 4)
  y <- matrix(rnorm(120), 60, 2)

  fit <- mlx_qr_gpu(as_mlx(x), as_mlx(y), method = "metal_householder")
  expected <- positive_base_qr(x, y)

  expect_equal(as.matrix(fit$R), expected$R, tolerance = 1e-4)
  expect_equal(as.matrix(fit$qty), expected$qty, tolerance = 1e-4)
  expect_equal(crossprod(as.matrix(fit$R)), crossprod(x), tolerance = 1e-4)
  expect_equal(fit$method, "metal_householder")
})

test_that("mlx_qr_gpu blocked Householder path matches base qr quantities", {
  skip_if_not(mlx_has_gpu())

  set.seed(20260613)
  x <- matrix(rnorm(420), 70, 6)
  y <- matrix(rnorm(140), 70, 2)

  fit <- mlx_qr_gpu(as_mlx(x), as_mlx(y), method = "blocked_householder")
  expected <- positive_base_qr(x, y)

  expect_equal(as.matrix(fit$R), expected$R, tolerance = 1e-4)
  expect_equal(as.matrix(fit$qty), expected$qty, tolerance = 1e-4)
  expect_equal(crossprod(as.matrix(fit$R)), crossprod(x), tolerance = 1e-4)
  expect_equal(fit$method, "blocked_householder")
})

test_that("mlx_qr_gpu custom Metal TSQR path matches base qr quantities", {
  skip_if_not(mlx_has_gpu())

  set.seed(20260613)
  x <- matrix(rnorm(180), 45, 4)
  y <- matrix(rnorm(90), 45, 2)

  fit <- mlx_qr_gpu(as_mlx(x), as_mlx(y), block_rows = 12L, method = "tsqr")
  expected <- positive_base_qr(x, y)

  expect_equal(as.matrix(fit$R), expected$R, tolerance = 1e-4)
  expect_equal(as.matrix(fit$qty), expected$qty, tolerance = 1e-4)
  expect_equal(crossprod(as.matrix(fit$R)), crossprod(x), tolerance = 1e-4)
  expect_equal(fit$method, "tsqr")
})

test_that("mlx_qr_gpu reports rank deficient inputs", {
  skip_if_not(mlx_has_gpu())

  set.seed(20260613)
  x <- matrix(rnorm(80), 20, 4)
  x[, 4] <- x[, 2]
  y <- matrix(rnorm(20), 20, 1)

  expect_error(
    mlx_qr_gpu(as_mlx(x), as_mlx(y), block_rows = 8L),
    "rank deficiency",
    fixed = TRUE
  )
  expect_error(
    mlx_qr_gpu(as_mlx(x), as_mlx(y), block_rows = 8L, method = "tsqr"),
    "rank deficiency",
    fixed = TRUE
  )
  expect_error(
    mlx_qr_gpu(as_mlx(x), as_mlx(y), method = "householder"),
    "rank deficiency",
    fixed = TRUE
  )
  expect_error(
    mlx_qr_gpu(as_mlx(x), as_mlx(y), method = "metal_householder"),
    "rank deficiency",
    fixed = TRUE
  )
  expect_error(
    mlx_qr_gpu(as_mlx(x), as_mlx(y), method = "blocked_householder"),
    "rank deficiency",
    fixed = TRUE
  )
})

test_that("mlx_qr_gpu rejects non-matrix inputs", {
  skip_if_not(mlx_has_gpu())

  expect_error(mlx_qr_gpu(as_mlx(1:4)), "2D matrix", fixed = TRUE)
  expect_error(mlx_qr_gpu(as_mlx(1)), "2D matrix", fixed = TRUE)
})
