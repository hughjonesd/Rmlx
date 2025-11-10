test_that("sqrt works", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  result <- as.matrix(sqrt(x_mlx))
  expect_equal(result, sqrt(x), tolerance = 1e-6)
})

test_that("exp and log work", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  result_exp <- as.matrix(exp(x_mlx))
  expect_equal(result_exp, exp(x), tolerance = 1e-6)

  result_log <- as.matrix(log(x_mlx))
  expect_equal(result_log, log(x), tolerance = 1e-6)
})

test_that("log2 and log10 work", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  result_log2 <- as.matrix(log2(x_mlx))
  expect_equal(result_log2, log2(x), tolerance = 1e-6)

  result_log10 <- as.matrix(log10(x_mlx))
  expect_equal(result_log10, log10(x), tolerance = 1e-6)
})

test_that("expm1 and log1p work", {
  x <- matrix(seq(0.01, 0.12, length.out = 12), 3, 4)
  x_mlx <- as_mlx(x)

  result_expm1 <- as.matrix(expm1(x_mlx))
  expect_equal(result_expm1, expm1(x), tolerance = 1e-6)

  result_log1p <- as.matrix(log1p(x_mlx))
  expect_equal(result_log1p, log1p(x), tolerance = 1e-6)
})

test_that("trig functions work", {
  x <- matrix(seq(-pi, pi, length.out = 12), 3, 4)
  x_mlx <- as_mlx(x)

  result_sin <- as.matrix(sin(x_mlx))
  expect_equal(result_sin, sin(x), tolerance = 1e-6)

  result_cos <- as.matrix(cos(x_mlx))
  expect_equal(result_cos, cos(x), tolerance = 1e-6)

  result_tan <- as.matrix(tan(x_mlx))
  expect_equal(result_tan, tan(x), tolerance = 1e-6)
})

test_that("inverse trig functions work", {
  x <- matrix(seq(-0.9, 0.9, length.out = 12), 3, 4)
  x_mlx <- as_mlx(x)

  result_asin <- as.matrix(asin(x_mlx))
  expect_equal(result_asin, asin(x), tolerance = 1e-6)

  result_acos <- as.matrix(acos(x_mlx))
  expect_equal(result_acos, acos(x), tolerance = 1e-6)

  result_atan <- as.matrix(atan(x_mlx))
  expect_equal(result_atan, atan(x), tolerance = 1e-6)
})

test_that("hyperbolic functions work", {
  x <- matrix(seq(-1, 1, length.out = 12), 3, 4)
  x_mlx <- as_mlx(x)

  result_sinh <- as.matrix(sinh(x_mlx))
  expect_equal(result_sinh, sinh(x), tolerance = 1e-6)

  result_cosh <- as.matrix(cosh(x_mlx))
  expect_equal(result_cosh, cosh(x), tolerance = 1e-6)

  result_tanh <- as.matrix(tanh(x_mlx))
  expect_equal(result_tanh, tanh(x), tolerance = 1e-6)
})

test_that("inverse hyperbolic functions work", {
  x <- matrix(seq(-0.9, 0.9, length.out = 12), 3, 4)
  x_mlx <- as_mlx(x)

  result_asinh <- as.matrix(asinh(x_mlx))
  expect_equal(result_asinh, asinh(x), tolerance = 1e-6)

  result_atanh <- as.matrix(atanh(x_mlx))
  expect_equal(result_atanh, atanh(x), tolerance = 1e-6)

  # acosh requires x >= 1
  x_acosh <- matrix(seq(1, 3, length.out = 12), 3, 4)
  x_acosh_mlx <- as_mlx(x_acosh)
  result_acosh <- as.matrix(acosh(x_acosh_mlx))
  expect_equal(result_acosh, acosh(x_acosh), tolerance = 1e-6)
})

test_that("rounding functions work", {
  x <- matrix(seq(-2.7, 2.7, length.out = 12), 3, 4)
  x_mlx <- as_mlx(x)

  result_floor <- as.matrix(floor(x_mlx))
  expect_equal(result_floor, floor(x), tolerance = 1e-6)

  result_ceiling <- as.matrix(ceiling(x_mlx))
  expect_equal(result_ceiling, ceiling(x), tolerance = 1e-6)

  result_round <- as.matrix(round(x_mlx))
  expect_equal(result_round, round(x), tolerance = 1e-6)

  result_round_digits <- as.matrix(round(x_mlx, digits = 2))
  expect_equal(result_round_digits, round(x, digits = 2), tolerance = 1e-6)
})

test_that("abs and sign work", {
  x <- matrix(seq(-6, 6, length.out = 12), 3, 4)
  x_mlx <- as_mlx(x)

  result_abs <- as.matrix(abs(x_mlx))
  expect_equal(result_abs, abs(x), tolerance = 1e-6)

  result_sign <- as.matrix(sign(x_mlx))
  expect_equal(result_sign, sign(x), tolerance = 1e-6)
})

test_that("cumulative operations work", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  # cumsum returns 1D mlx array
  result_cumsum <- as.vector(cumsum(x_mlx))
  expect_equal(result_cumsum, cumsum(x), tolerance = 1e-6)

  # cumprod
  result_cumprod <- as.vector(cumprod(x_mlx))
  expect_equal(result_cumprod, cumprod(x), tolerance = 1e-6)

  # cummax
  x2 <- matrix(c(5, 2, 8, 3, 1, 9, 4, 7, 6, 10, 11, 12), 3, 4)
  x2_mlx <- as_mlx(x2)
  result_cummax <- as.vector(cummax(x2_mlx))
  expect_equal(result_cummax, cummax(x2), tolerance = 1e-6)

  # cummin
  result_cummin <- as.vector(cummin(x2_mlx))
  expect_equal(result_cummin, cummin(x2), tolerance = 1e-6)
})

test_that("mlx_degrees and mlx_radians convert angles", {
  radians <- as_mlx(c(0, pi / 2, pi))
  degrees <- mlx_degrees(radians)
  expect_equal(as.numeric(as.matrix(degrees)), c(0, 90, 180), tolerance = 1e-6)

  deg_values <- as_mlx(c(0, 180, 270))
  rad_out <- mlx_radians(deg_values)
  expect_equal(as.numeric(as.matrix(rad_out)), c(0, pi, 3 * pi / 2), tolerance = 1e-6)
})

test_that("mlx_nan/inf helpers work", {
  x <- as_mlx(c(-Inf, -1, NaN, 0, Inf))

  expect_equal(as.vector(mlx_isposinf(x)), c(FALSE, FALSE, FALSE, FALSE, TRUE))
  expect_equal(as.vector(mlx_isneginf(x)), c(TRUE, FALSE, FALSE, FALSE, FALSE))
  expect_equal(as.vector(mlx_isnan(x)), c(FALSE, FALSE, TRUE, FALSE, FALSE))
  expect_equal(as.vector(mlx_isinf(x)), c(TRUE, FALSE, FALSE, FALSE, TRUE))
  expect_equal(as.vector(mlx_isfinite(x)), c(FALSE, TRUE, FALSE, TRUE, FALSE))

  replaced <- mlx_nan_to_num(x, nan = 0, posinf = 10, neginf = -10)
  expect_equal(as.numeric(as.matrix(replaced)), c(-10, -1, 0, 0, 10), tolerance = 1e-6)
})

test_that("is.nan/is.infinite/is.finite methods dispatch", {
  x <- as_mlx(c(-Inf, -1, NaN, 0, Inf))

  expect_s3_class(is.nan(x), "mlx")
  expect_equal(as.vector(is.nan(x)), c(FALSE, FALSE, TRUE, FALSE, FALSE))

  expect_s3_class(is.infinite(x), "mlx")
  expect_equal(as.vector(is.infinite(x)), c(TRUE, FALSE, FALSE, FALSE, TRUE))

  expect_s3_class(is.finite(x), "mlx")
  expect_equal(as.vector(is.finite(x)), c(FALSE, TRUE, FALSE, TRUE, FALSE))
})

test_that("fft matches base R", {
  set.seed(123)
  x <- rnorm(16)
  x_mlx <- as_mlx(x)

  fft_r <- fft(x)
  fft_mlx <- fft(x_mlx)

  expect_s3_class(fft_mlx, "mlx")
  expect_equal(as.vector(fft_mlx), fft_r, tolerance = 1e-4)

  ifft_mlx <- fft(fft_mlx, inverse = TRUE)
  ifft_r <- fft(fft_r, inverse = TRUE)

  expect_equal(as.vector(ifft_mlx), ifft_r, tolerance = 1e-4)
})

test_that("unsupported Math functions fall back to R", {
  x <- matrix(seq(-2.7, 2.7, length.out = 12), 3, 4)
  x_mlx <- as_mlx(x)

  # trunc not in MLX, should fall back to R with warning
  expect_warning(
    result_trunc <- as.matrix(trunc(x_mlx)),
    "MLX does not support 'trunc'"
  )
  expect_equal(result_trunc, trunc(x), tolerance = 1e-6)

  # gamma not in MLX, should fall back to R with warning
  x_pos <- matrix(seq(0.5, 3.5, length.out = 12), 3, 4)
  x_pos_mlx <- as_mlx(x_pos)
  expect_warning(
    result_gamma <- as.matrix(gamma(x_pos_mlx)),
    "MLX does not support 'gamma'"
  )
  expect_equal(result_gamma, gamma(x_pos), tolerance = 1e-6)

  # lgamma not in MLX, should fall back to R with warning
  expect_warning(
    result_lgamma <- as.matrix(lgamma(x_pos_mlx)),
    "MLX does not support 'lgamma'"
  )
  expect_equal(result_lgamma, lgamma(x_pos), tolerance = 1e-6)
})

test_that("complex helpers and generics operate on mlx arrays", {
  arr <- matrix(1:4, 2, 2)
  z <- as_mlx(arr + 1i * (arr + 1))

  re_part <- mlx_real(z)
  im_part <- mlx_imag(z)
  conj_part <- mlx_conjugate(z)

  expect_equal(as.matrix(re_part), arr, tolerance = 1e-6)
  expect_equal(as.matrix(im_part), arr + 1, tolerance = 1e-6)
  expect_equal(as.matrix(conj_part), Conj(arr + 1i * (arr + 1)), tolerance = 1e-6)

  expect_equal(as.matrix(Re(z)), arr, tolerance = 1e-6)
  expect_equal(as.matrix(Im(z)), arr + 1, tolerance = 1e-6)
  expect_equal(as.matrix(Conj(z)), Conj(arr + 1i * (arr + 1)), tolerance = 1e-6)
})

test_that("mlx_isclose returns element-wise comparison", {
  a <- as_mlx(c(1.0, 2.0, 3.0))
  b <- as_mlx(c(1.0 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-3))

  # Default tolerance should pass for small differences
  result <- mlx_isclose(a, b)
  expect_s3_class(result, "mlx")
  close_vals <- as.vector(result)
  expect_equal(close_vals[1:2], c(TRUE, TRUE))
  expect_equal(close_vals[3], FALSE)  # 1e-3 is too large

  # Tighter tolerance
  result_strict <- mlx_isclose(a, b, rtol = 1e-7, atol = 1e-9)
  close_strict <- as.vector(result_strict)
  expect_equal(close_strict, c(FALSE, FALSE, FALSE))

  # Broadcasting
  a_mat <- as_mlx(matrix(1:6, 2, 3))
  b_scalar <- as_mlx(3.0)
  result_bcast <- mlx_isclose(a_mat, b_scalar)
  expect_equal(dim(as.matrix(result_bcast)), c(2L, 3L))
})

test_that("mlx_allclose returns scalar boolean", {
  a <- as_mlx(c(1.0, 2.0, 3.0))
  b <- as_mlx(c(1.0 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-6))

  # All elements close with default tolerance
  result <- mlx_allclose(a, b)
  expect_s3_class(result, "mlx")
  expect_true(as.logical(as.matrix(result)))

  # Not all close with stricter tolerance
  result_strict <- mlx_allclose(a, b, rtol = 1e-7, atol = 1e-9)
  expect_false(as.logical(as.matrix(result_strict)))

  # Test with one element far off
  c <- as_mlx(c(1.0, 2.0, 100.0))
  result_diff <- mlx_allclose(a, c)
  expect_false(as.logical(as.matrix(result_diff)))
})

test_that("mlx_isclose handles NaN with equal_nan parameter", {
  a <- as_mlx(c(1.0, NaN, 3.0))
  b <- as_mlx(c(1.0, NaN, 3.0))

  # By default, NaN != NaN
  result_default <- mlx_isclose(a, b)
  close_vals <- as.vector(result_default)
  expect_equal(close_vals[1], TRUE)
  expect_equal(close_vals[2], FALSE)  # NaN not equal to NaN
  expect_equal(close_vals[3], TRUE)

  # With equal_nan = TRUE
  result_equal_nan <- mlx_isclose(a, b, equal_nan = TRUE)
  close_vals_nan <- as.vector(result_equal_nan)
  expect_equal(close_vals_nan, c(TRUE, TRUE, TRUE))
})

test_that("all.equal.mlx follows R semantics", {
  a <- as_mlx(c(1.0, 2.0, 3.0))
  b <- as_mlx(c(1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9))

  # Should return TRUE when all close (within default tolerance)
  result <- all.equal(a, b)
  expect_true(isTRUE(result))

  # Should return character vector describing differences when not close
  c <- as_mlx(c(1.0, 2.0, 10.0))
  result_diff <- all.equal(a, c)
  expect_type(result_diff, "character")
  expect_match(result_diff, "not all close", ignore.case = TRUE)

  # Test with tolerance parameter
  d <- as_mlx(c(1.0, 2.0, 3.01))
  result_tol <- all.equal(a, d, tolerance = 0.02)
  expect_true(isTRUE(result_tol))

  result_notol <- all.equal(a, d, tolerance = 0.001)
  expect_type(result_notol, "character")

  # Test shape mismatch
  e <- as_mlx(c(1.0, 2.0))
  result_shape <- all.equal(a, e)
  expect_type(result_shape, "character")
  expect_match(result_shape, "shape|length|dim", ignore.case = TRUE)
})

test_that("mlx_erf works", {
  x <- c(-2, -1, 0, 1, 2)
  x_mlx <- as_mlx(x)

  result <- as.vector(mlx_erf(x_mlx))
  # R doesn't have erf, but we can use 2*pnorm(x*sqrt(2)) - 1
  expected <- 2 * pnorm(x * sqrt(2)) - 1
  expect_equal(result, expected, tolerance = 1e-6)
})

test_that("mlx_erfinv works", {
  p <- c(-0.5, 0, 0.5)
  p_mlx <- as_mlx(p)

  result <- as.vector(mlx_erfinv(p_mlx))
  # erfinv should be the inverse of erf
  # erf(erfinv(p)) = p
  back <- as.vector(mlx_erf(as_mlx(result)))
  expect_equal(back, p, tolerance = 1e-6)
})

test_that("mlx_dnorm works", {
  x <- seq(-3, 3, by = 0.5)
  x_mlx <- as_mlx(x)

  # Standard normal
  result <- as.vector(mlx_dnorm(x_mlx))
  expected <- dnorm(x)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard normal
  result_nonstd <- as.vector(mlx_dnorm(x_mlx, mean = 1, sd = 2))
  expected_nonstd <- dnorm(x, mean = 1, sd = 2)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)

  # Log density
  result_log <- as.vector(mlx_dnorm(x_mlx, log = TRUE))
  expected_log <- dnorm(x, log = TRUE)
  expect_equal(result_log, expected_log, tolerance = 1e-6)
})

test_that("mlx_pnorm works", {
  x <- seq(-3, 3, by = 0.5)
  x_mlx <- as_mlx(x)

  # Standard normal
  result <- as.vector(mlx_pnorm(x_mlx))
  expected <- pnorm(x)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard normal
  result_nonstd <- as.vector(mlx_pnorm(x_mlx, mean = 1, sd = 2))
  expected_nonstd <- pnorm(x, mean = 1, sd = 2)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)
})

test_that("mlx_qnorm works", {
  p <- c(0.025, 0.25, 0.5, 0.75, 0.975)
  p_mlx <- as_mlx(p)

  # Standard normal
  result <- as.vector(mlx_qnorm(p_mlx))
  expected <- qnorm(p)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard normal
  result_nonstd <- as.vector(mlx_qnorm(p_mlx, mean = 1, sd = 2))
  expected_nonstd <- qnorm(p, mean = 1, sd = 2)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)

  # Round trip: qnorm(pnorm(x)) = x
  x <- seq(-2, 2, by = 0.5)
  x_mlx <- as_mlx(x)
  p_result <- mlx_pnorm(x_mlx)
  back <- as.vector(mlx_qnorm(p_result))
  expect_equal(back, x, tolerance = 1e-6)
})

test_that("mlx_dunif works", {
  x <- seq(-0.5, 1.5, by = 0.1)
  x_mlx <- as_mlx(x)

  # Standard uniform
  result <- as.vector(mlx_dunif(x_mlx))
  expected <- dunif(x)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard uniform
  result_nonstd <- as.vector(mlx_dunif(x_mlx, min = -1, max = 2))
  expected_nonstd <- dunif(x, min = -1, max = 2)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)

  # Log density
  result_log <- as.vector(mlx_dunif(x_mlx, log = TRUE))
  expected_log <- dunif(x, log = TRUE)
  expect_equal(result_log, expected_log, tolerance = 1e-6)
})

test_that("mlx_punif works", {
  x <- seq(-0.5, 1.5, by = 0.1)
  x_mlx <- as_mlx(x)

  # Standard uniform
  result <- as.vector(mlx_punif(x_mlx))
  expected <- punif(x)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard uniform
  result_nonstd <- as.vector(mlx_punif(x_mlx, min = -1, max = 2))
  expected_nonstd <- punif(x, min = -1, max = 2)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)
})

test_that("mlx_qunif works", {
  p <- seq(0, 1, by = 0.1)
  p_mlx <- as_mlx(p)

  # Standard uniform
  result <- as.vector(mlx_qunif(p_mlx))
  expected <- qunif(p)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard uniform
  result_nonstd <- as.vector(mlx_qunif(p_mlx, min = -1, max = 2))
  expected_nonstd <- qunif(p, min = -1, max = 2)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)

  # Round trip
  x <- seq(0, 1, by = 0.1)
  x_mlx <- as_mlx(x)
  p_result <- mlx_punif(x_mlx)
  back <- as.vector(mlx_qunif(p_result))
  expect_equal(back, x, tolerance = 1e-6)
})

test_that("mlx_dexp works", {
  x <- seq(0, 5, by = 0.5)
  x_mlx <- as_mlx(x)

  # Standard exponential
  result <- as.vector(mlx_dexp(x_mlx))
  expected <- dexp(x)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard rate
  result_nonstd <- as.vector(mlx_dexp(x_mlx, rate = 2))
  expected_nonstd <- dexp(x, rate = 2)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)

  # Log density
  result_log <- as.vector(mlx_dexp(x_mlx, log = TRUE))
  expected_log <- dexp(x, log = TRUE)
  expect_equal(result_log, expected_log, tolerance = 1e-6)

  # Negative values should be 0
  x_neg <- as_mlx(c(-1, -0.5, 0, 0.5, 1))
  result_neg <- as.vector(mlx_dexp(x_neg))
  expected_neg <- dexp(c(-1, -0.5, 0, 0.5, 1))
  expect_equal(result_neg, expected_neg, tolerance = 1e-6)
})

test_that("mlx_pexp works", {
  x <- seq(0, 5, by = 0.5)
  x_mlx <- as_mlx(x)

  # Standard exponential
  result <- as.vector(mlx_pexp(x_mlx))
  expected <- pexp(x)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard rate
  result_nonstd <- as.vector(mlx_pexp(x_mlx, rate = 2))
  expected_nonstd <- pexp(x, rate = 2)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)
})

test_that("mlx_qexp works", {
  p <- seq(0.1, 0.9, by = 0.1)
  p_mlx <- as_mlx(p)

  # Standard exponential
  result <- as.vector(mlx_qexp(p_mlx))
  expected <- qexp(p)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard rate
  result_nonstd <- as.vector(mlx_qexp(p_mlx, rate = 2))
  expected_nonstd <- qexp(p, rate = 2)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)

  # Round trip
  x <- seq(0.5, 5, by = 0.5)
  x_mlx <- as_mlx(x)
  p_result <- mlx_pexp(x_mlx)
  back <- as.vector(mlx_qexp(p_result))
  expect_equal(back, x, tolerance = 1e-6)
})

test_that("mlx_dlnorm works", {
  x <- seq(0.1, 3, by = 0.2)
  x_mlx <- as_mlx(x)

  # Standard lognormal
  result <- as.vector(mlx_dlnorm(x_mlx))
  expected <- dlnorm(x)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard lognormal
  result_nonstd <- as.vector(mlx_dlnorm(x_mlx, meanlog = 1, sdlog = 0.5))
  expected_nonstd <- dlnorm(x, meanlog = 1, sdlog = 0.5)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)

  # Log density
  result_log <- as.vector(mlx_dlnorm(x_mlx, log = TRUE))
  expected_log <- dlnorm(x, log = TRUE)
  expect_equal(result_log, expected_log, tolerance = 1e-6)
})

test_that("mlx_plnorm works", {
  x <- seq(0.1, 3, by = 0.2)
  x_mlx <- as_mlx(x)

  # Standard lognormal
  result <- as.vector(mlx_plnorm(x_mlx))
  expected <- plnorm(x)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard lognormal
  result_nonstd <- as.vector(mlx_plnorm(x_mlx, meanlog = 1, sdlog = 0.5))
  expected_nonstd <- plnorm(x, meanlog = 1, sdlog = 0.5)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)
})

test_that("mlx_qlnorm works", {
  p <- seq(0.1, 0.9, by = 0.1)
  p_mlx <- as_mlx(p)

  # Standard lognormal
  result <- as.vector(mlx_qlnorm(p_mlx))
  expected <- qlnorm(p)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard lognormal
  result_nonstd <- as.vector(mlx_qlnorm(p_mlx, meanlog = 1, sdlog = 0.5))
  expected_nonstd <- qlnorm(p, meanlog = 1, sdlog = 0.5)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)

  # Round trip
  x <- seq(0.5, 3, by = 0.25)
  x_mlx <- as_mlx(x)
  p_result <- mlx_plnorm(x_mlx)
  back <- as.vector(mlx_qlnorm(p_result))
  expect_equal(back, x, tolerance = 1e-6)
})

test_that("mlx_dlogis works", {
  x <- seq(-3, 3, by = 0.5)
  x_mlx <- as_mlx(x)

  # Standard logistic
  result <- as.vector(mlx_dlogis(x_mlx))
  expected <- dlogis(x)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard logistic
  result_nonstd <- as.vector(mlx_dlogis(x_mlx, location = 1, scale = 2))
  expected_nonstd <- dlogis(x, location = 1, scale = 2)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)

  # Log density
  result_log <- as.vector(mlx_dlogis(x_mlx, log = TRUE))
  expected_log <- dlogis(x, log = TRUE)
  expect_equal(result_log, expected_log, tolerance = 1e-6)
})

test_that("mlx_plogis works", {
  x <- seq(-3, 3, by = 0.5)
  x_mlx <- as_mlx(x)

  # Standard logistic
  result <- as.vector(mlx_plogis(x_mlx))
  expected <- plogis(x)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard logistic
  result_nonstd <- as.vector(mlx_plogis(x_mlx, location = 1, scale = 2))
  expected_nonstd <- plogis(x, location = 1, scale = 2)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)
})

test_that("mlx_qlogis works", {
  p <- seq(0.1, 0.9, by = 0.1)
  p_mlx <- as_mlx(p)

  # Standard logistic
  result <- as.vector(mlx_qlogis(p_mlx))
  expected <- qlogis(p)
  expect_equal(result, expected, tolerance = 1e-6)

  # Non-standard logistic
  result_nonstd <- as.vector(mlx_qlogis(p_mlx, location = 1, scale = 2))
  expected_nonstd <- qlogis(p, location = 1, scale = 2)
  expect_equal(result_nonstd, expected_nonstd, tolerance = 1e-6)

  # Round trip
  x <- seq(-2, 2, by = 0.5)
  x_mlx <- as_mlx(x)
  p_result <- mlx_plogis(x_mlx)
  back <- as.vector(mlx_qlogis(p_result))
  expect_equal(back, x, tolerance = 1e-6)
})
