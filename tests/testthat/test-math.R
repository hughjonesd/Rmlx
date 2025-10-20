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
})

test_that("abs and sign work", {
  x <- matrix(seq(-6, 6, length.out = 12), 3, 4)
  x_mlx <- as_mlx(x)

  result_abs <- as.matrix(abs(x_mlx))
  expect_equal(result_abs, abs(x), tolerance = 1e-6)

  result_sign <- as.matrix(sign(x_mlx))
  expect_equal(result_sign, sign(x), tolerance = 1e-6)
})
