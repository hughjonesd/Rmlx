skip_on_cran()

set.seed(42)

expect_fft_equal <- function(x, y, tol = 1e-6) {
  res <- as.vector(x)
  expect_equal(res, as.complex(as.vector(y)), tolerance = tol)
}

test_that("mlx_fft matches base fft for vectors", {
  x <- 1:8
  res <- mlx_fft(x)
  expect_fft_equal(res, stats::fft(x))

  inv <- mlx_fft(res, inverse = TRUE)
  expect_fft_equal(inv, stats::fft(stats::fft(x), inverse = TRUE))
})

test_that("mlx_fft axis argument works", {
  arr <- array(1:12, dim = c(3, 4))
  res <- mlx_fft(arr, axis = 1L)
  ref <- apply(arr, 2, stats::fft)
  expect_equal(as.matrix(res), ref, tolerance = 1e-6)
})

test_that("mlx_fft2 round-trips matrices", {
  mat <- matrix(runif(9), 3, 3)
  res <- mlx_fft2(mat)
  inv <- mlx_fft2(res, inverse = TRUE)
  inv_scaled <- as.matrix(inv) / prod(dim(mat))
  expect_equal(Re(inv_scaled), mat, tolerance = 1e-6)
  expect_lt(max(abs(Im(inv_scaled))), 1e-6)
})

test_that("mlx_fftn round-trips arrays", {
  arr <- array(runif(16), dim = c(2, 2, 4))
  res <- mlx_fftn(arr)
  inv <- mlx_fftn(res, inverse = TRUE)
  inv_scaled <- as.array(inv) / prod(dim(arr))
  expect_equal(Re(inv_scaled), arr, tolerance = 1e-6)
  expect_lt(max(abs(Im(inv_scaled))), 1e-6)
})

test_that("mlx_fftn supports custom axes", {
  arr <- array(runif(24), dim = c(2, 3, 4))
  res <- mlx_fftn(arr, axes = c(1, 3))
  inv <- mlx_fftn(res, axes = c(1, 3), inverse = TRUE)
  axes_lengths <- dim(arr)[c(1, 3)]
  inv_scaled <- as.array(inv) / prod(axes_lengths)
  expect_equal(Re(inv_scaled), arr, tolerance = 1e-6)
  expect_lt(max(abs(Im(inv_scaled))), 1e-6)
})
