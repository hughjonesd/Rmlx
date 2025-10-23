skip_on_cran()

test_that("mlx_rand_uniform generates correct shape and values", {
  set.seed(123)
  shape <- c(256L, 256L)
  mx_rng <- mlx_rand_uniform(shape, min = -1, max = 1)

  expect_s3_class(mx_rng, "mlx")
  expect_equal(mx_rng$dim, shape)
  vals <- as.matrix(mx_rng)
  expect_true(all(vals >= -1 & vals <= 1))
})

test_that("mlx_rand_normal generates correct distribution", {
  set.seed(123)
  shape <- c(256L, 256L)
  mx_norm <- mlx_rand_normal(shape, mean = 2, sd = 0.5)

  expect_s3_class(mx_norm, "mlx")
  expect_equal(mx_norm$dim, shape)
  norm_vals <- as.matrix(mx_norm)
  expect_true(abs(mean(norm_vals) - 2) < 0.1)
  expect_true(sd(norm_vals) > 0)
})

test_that("mlx_rand_bernoulli generates binary values", {
  set.seed(123)
  shape <- c(256L, 256L)
  mx_bern <- mlx_rand_bernoulli(shape, prob = 0.3)

  expect_s3_class(mx_bern, "mlx")
  expect_equal(mx_bern$dim, shape)
  bern_vals <- as.matrix(mx_bern)
  expect_true(all(bern_vals %in% c(0, 1)))
})

test_that("mlx_rand_gumbel generates correct distribution", {
  set.seed(123)
  shape <- c(256L, 256L)
  mx_gumbel <- mlx_rand_gumbel(shape)

  expect_s3_class(mx_gumbel, "mlx")
  expect_equal(mx_gumbel$dim, shape)
  gumbel_vals <- as.matrix(mx_gumbel)
  expect_true(all(is.finite(gumbel_vals)))
  # Gumbel distribution mean should be close to Euler-Mascheroni constant (~0.5772)
  expect_true(abs(mean(gumbel_vals) - 0.5772) < 0.05)
  # Standard deviation should be close to pi/sqrt(6) (~1.2825)
  expect_true(abs(sd(gumbel_vals) - 1.2825) < 0.1)
})

test_that("mlx_rand_truncated_normal generates values within bounds", {
  set.seed(123)
  mx_tnorm <- mlx_rand_truncated_normal(-1, 1, c(100L, 100L))

  expect_s3_class(mx_tnorm, "mlx")
  expect_equal(mx_tnorm$dim, c(100L, 100L))
  tnorm_vals <- as.matrix(mx_tnorm)
  expect_true(all(tnorm_vals >= -1 & tnorm_vals <= 1))
  expect_true(all(is.finite(tnorm_vals)))
})

test_that("mlx_rand_truncated_normal works with different bounds", {
  set.seed(123)
  mx_tnorm <- mlx_rand_truncated_normal(0, 10, c(50L, 50L))
  tnorm_vals <- as.matrix(mx_tnorm)

  expect_true(all(tnorm_vals >= 0 & tnorm_vals <= 10))
})

test_that("mlx_rand_truncated_normal works on CPU device", {
  set.seed(123)
  mx_tnorm <- mlx_rand_truncated_normal(-2, 2, c(40L, 40L), device = "cpu")

  expect_equal(mx_tnorm$device, "cpu")
  expect_equal(mx_tnorm$dim, c(40L, 40L))
})

test_that("mlx_rand_multivariate_normal generates finite values", {
  set.seed(123)
  mvn_mean <- as_mlx(c(0, 0), device = "cpu")
  mvn_cov <- as_mlx(matrix(c(1, 0, 0, 1), 2, 2), device = "cpu")
  mx_mvn <- mlx_rand_multivariate_normal(c(10L, 2L), mvn_mean, mvn_cov, device = "cpu")

  expect_s3_class(mx_mvn, "mlx")
  expect_equal(mx_mvn$device, "cpu")
  expect_equal(mx_mvn$dtype, "float32")
  mvn_vals <- as.vector(as.matrix(mx_mvn))
  expect_true(all(is.finite(mvn_vals)))
})

test_that("mlx_rand_multivariate_normal works with non-identity covariance", {
  set.seed(123)
  mvn_mean <- as_mlx(c(1, 2), device = "cpu")
  mvn_cov <- as_mlx(matrix(c(2, 0.5, 0.5, 1), 2, 2), device = "cpu")
  mx_mvn <- mlx_rand_multivariate_normal(c(5L, 2L), mvn_mean, mvn_cov, device = "cpu")

  expect_s3_class(mx_mvn, "mlx")
  expect_equal(mx_mvn$device, "cpu")
  mvn_vals <- as.vector(as.matrix(mx_mvn))
  expect_true(all(is.finite(mvn_vals)))
})

test_that("mlx_rand_laplace generates correct shape and values", {
  set.seed(123)
  shape <- c(256L, 256L)
  mx_laplace <- mlx_rand_laplace(shape, loc = 0, scale = 1)

  expect_s3_class(mx_laplace, "mlx")
  expect_equal(mx_laplace$dim, shape)
  laplace_vals <- as.matrix(mx_laplace)
  expect_true(all(is.finite(laplace_vals)))
  # Mean should be close to loc
  expect_true(abs(mean(laplace_vals) - 0) < 0.1)
})

test_that("mlx_rand_laplace works with different parameters", {
  set.seed(123)
  mx_laplace <- mlx_rand_laplace(c(100L, 100L), loc = 5, scale = 2)

  expect_s3_class(mx_laplace, "mlx")
  laplace_vals <- as.matrix(mx_laplace)
  expect_true(all(is.finite(laplace_vals)))
  # Mean should be close to loc
  expect_true(abs(mean(laplace_vals) - 5) < 0.3)
})

test_that("mlx_rand_categorical generates valid indices", {
  set.seed(123)
  # Simple categorical with 3 classes
  logits <- as_mlx(matrix(c(0.5, 0.2, 0.3), 1, 3))
  samples <- mlx_rand_categorical(logits, num_samples = 100)

  expect_s3_class(samples, "mlx")
  expect_equal(samples$dtype, "int32")
  sample_vals <- as.vector(as.matrix(samples))
  # Indices should be in valid range [0, 2] for 3 classes
  expect_true(all(sample_vals >= 0 & sample_vals < 3))
})

test_that("mlx_rand_categorical works with multiple rows", {
  set.seed(123)
  # Multiple categorical distributions
  logits <- as_mlx(matrix(c(1, 2, 3, 3, 2, 1), 2, 3, byrow = TRUE))
  samples <- mlx_rand_categorical(logits, axis = -1L, num_samples = 10)

  expect_s3_class(samples, "mlx")
  sample_vals <- as.matrix(samples)
  # Indices should be in valid range [0, 2] for 3 classes
  expect_true(all(sample_vals >= 0 & sample_vals < 3))
})
