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
