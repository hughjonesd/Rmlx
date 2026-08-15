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

test_that("mlx_qr_gpu CholeskyQR2 improves an ill-conditioned solve", {
  skip_if_not(mlx_has_gpu())

  set.seed(20260815)
  n <- 20000L
  p <- 8L
  x <- matrix(rnorm(n * p), n, p)
  x[, p] <- x[, 1L] + 1e-3 * rnorm(n)
  y <- matrix(rnorm(n), n, 1L)
  expected <- qr.solve(x, y)

  once <- mlx_qr_gpu(as_mlx(x), as_mlx(y), tol = 1e-6,
                     method = "cholqr")
  twice <- mlx_qr_gpu(as_mlx(x), as_mlx(y), tol = 1e-6,
                      method = "cholqr2")
  coef_once <- as.matrix(mlx_solve_triangular(
    once$R, once$qty, upper = TRUE, device = "cpu"
  ))
  coef_twice <- as.matrix(mlx_solve_triangular(
    twice$R, twice$qty, upper = TRUE, device = "cpu"
  ))

  error_once <- sqrt(sum((coef_once - expected)^2) / sum(expected^2))
  error_twice <- sqrt(sum((coef_twice - expected)^2) / sum(expected^2))
  expect_lt(error_twice, error_once / 10)
  expect_lt(error_twice, 1e-3)
  expect_equal(twice$method, "cholqr2")
})

test_that("mlx_qr_gpu CholeskyQR2 corrects its least-squares quantities", {
  skip_if_not(mlx_has_gpu())

  set.seed(9300)
  n <- 5000L
  p <- 200L
  x <- cbind(1, matrix(rnorm(n * (p - 1L)), n, p - 1L))
  beta <- seq_len(p) / p
  y <- matrix(drop(x %*% beta) + rnorm(n), n, 1L)

  fit <- mlx_qr_gpu(as_mlx(x), as_mlx(y), method = "cholqr2")
  expected <- positive_base_qr(x, y)
  actual <- mlx_solve_triangular(
    fit$R, fit$qty_corrected, upper = TRUE, device = "cpu"
  )

  expect_equal(as.matrix(fit$qty), expected$qty, tolerance = 1e-4)
  expect_equal(
    drop(as.matrix(actual)),
    unname(lm.fit(x, y)$coefficients),
    tolerance = 1e-7
  )
})

test_that("mlx_qr_gpu CholeskyQR2 falls back when its first pass is unsafe", {
  skip_if_not(mlx_has_gpu())

  set.seed(20260816)
  n <- 10000L
  p <- 6L
  x <- matrix(rnorm(n * p), n, p)
  x[, p] <- x[, 1L] + 2e-4 * rnorm(n)
  y <- matrix(rnorm(n), n, 1L)

  fit <- mlx_qr_gpu(as_mlx(x), as_mlx(y), tol = 1e-7,
                    method = "cholqr2")

  expect_equal(fit$requested_method, "cholqr2")
  expect_true(fit$method %in% c("tsqr", "cpu_qr"))
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

test_that("mlx_qr_gpu TSQR chooses a GPU tile size automatically", {
  skip_if_not(mlx_has_gpu())

  set.seed(20260815)
  x <- matrix(rnorm(127 * 8), 127, 8)
  y <- matrix(rnorm(127 * 3), 127, 3)

  fit <- mlx_qr_gpu(as_mlx(x), as_mlx(y), method = "tsqr")
  expected <- positive_base_qr(x, y)

  expect_lte(fit$block_rows * (ncol(x) + ncol(y)) * 4L, 32768L)
  expect_false(identical(fit$block_rows, 2048L))
  expect_equal(as.matrix(fit$R), expected$R, tolerance = 1e-4)
  expect_equal(as.matrix(fit$qty), expected$qty, tolerance = 1e-4)
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
    mlx_qr_gpu(as_mlx(x), as_mlx(y), block_rows = 8L,
               method = "cholqr2"),
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
