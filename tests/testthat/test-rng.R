skip_on_cran()

test_that("mlx_rand_uniform generates correct shape and values", {
  set.seed(123)
  shape <- c(256L, 256L)
  mx_rng <- mlx_rand_uniform(shape, min = -1, max = 1)

  expect_s3_class(mx_rng, "mlx")
  expect_equal(dim(mx_rng), shape)
  vals <- as.matrix(mx_rng)
  expect_true(all(vals >= -1 & vals <= 1))
})

test_that("mlx_rand_normal generates correct distribution", {
  set.seed(123)
  shape <- c(256L, 256L)
  mx_norm <- mlx_rand_normal(shape, mean = 2, sd = 0.5)

  expect_s3_class(mx_norm, "mlx")
  expect_equal(dim(mx_norm), shape)
  norm_vals <- as.matrix(mx_norm)
  expect_true(abs(mean(norm_vals) - 2) < 0.1)
  expect_true(sd(norm_vals) > 0)
})

test_that("mlx_rand_bernoulli generates binary values", {
  set.seed(123)
  shape <- c(256L, 256L)
  mx_bern <- mlx_rand_bernoulli(shape, prob = 0.3)

  expect_s3_class(mx_bern, "mlx")
  expect_equal(dim(mx_bern), shape)
  bern_vals <- as.matrix(mx_bern)
  expect_true(all(bern_vals %in% c(0, 1)))
})

test_that("mlx_rand_gumbel generates correct distribution", {
  set.seed(123)
  shape <- c(256L, 256L)
  mx_gumbel <- mlx_rand_gumbel(shape)

  expect_s3_class(mx_gumbel, "mlx")
  expect_equal(dim(mx_gumbel), shape)
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
  expect_equal(dim(mx_tnorm), c(100L, 100L))
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
  expect_equal(dim(mx_tnorm), c(40L, 40L))
})

test_that("mlx_rand_multivariate_normal generates finite values", {
  set.seed(123)
  mvn_mean <- as_mlx(c(0, 0), device = "cpu")
  mvn_cov <- as_mlx(matrix(c(1, 0, 0, 1), 2, 2), device = "cpu")
  mx_mvn <- mlx_rand_multivariate_normal(c(10L, 2L), mvn_mean, mvn_cov, device = "cpu")

  expect_s3_class(mx_mvn, "mlx")
  expect_equal(mx_mvn$device, "cpu")
  expect_equal(mlx_dtype(mx_mvn), "float32")
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
  expect_equal(dim(mx_laplace), shape)
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
  expect_equal(mlx_dtype(samples), "int32")
  sample_vals <- as.vector(as.matrix(samples))
  # Indices should be in valid range [1, 3] for 3 classes
  expect_true(all(sample_vals >= 1 & sample_vals <= 3))
})

test_that("mlx_rand_categorical works with multiple rows", {
  set.seed(123)
  # Multiple categorical distributions
  logits <- as_mlx(matrix(c(1, 2, 3, 3, 2, 1), 2, 3, byrow = TRUE))
  samples <- mlx_rand_categorical(logits, axis = 2L, num_samples = 10)

  expect_s3_class(samples, "mlx")
  sample_vals <- as.matrix(samples)
  # Indices should be in valid range [1, 3] for 3 classes
  expect_true(all(sample_vals >= 1 & sample_vals <= 3))
})

test_that("mlx_rand_randint generates integers in range", {
  set.seed(123)
  samples <- mlx_rand_randint(c(100L, 100L), low = 0, high = 10)

  expect_s3_class(samples, "mlx")
  expect_equal(dim(samples), c(100L, 100L))
  expect_equal(mlx_dtype(samples), "int32")
  sample_vals <- as.matrix(samples)
  expect_true(all(sample_vals >= 0 & sample_vals < 10))
})

test_that("mlx_rand_randint works with negative range", {
  set.seed(123)
  samples <- mlx_rand_randint(c(50L, 50L), low = -5, high = 5)

  expect_s3_class(samples, "mlx")
  sample_vals <- as.matrix(samples)
  expect_true(all(sample_vals >= -5 & sample_vals < 5))
})

test_that("mlx_rand_randint works with different dtypes", {
  set.seed(123)
  samples64 <- mlx_rand_randint(c(10L, 10L), low = 0, high = 100, dtype = "int64")

  expect_s3_class(samples64, "mlx")
  expect_equal(mlx_dtype(samples64), "int64")
})

test_that("mlx_rand_permutation generates valid permutation", {
  set.seed(123)
  perm <- mlx_rand_permutation(10)

  expect_s3_class(perm, "mlx")
  expect_equal(mlx_dtype(perm), "int32")
  expect_equal(length(dim(perm)), 1)
  expect_equal(dim(perm), 10L)

  perm_vals <- as.vector(perm)
  # Should contain each of 1:10 exactly once
  expect_equal(sort(perm_vals), 1:10)
})

test_that("mlx_rand_permutation permutes array rows", {
  set.seed(123)
  mat <- matrix(1:12, 4, 3)
  perm_mat <- mlx_rand_permutation(mat)

  expect_s3_class(perm_mat, "mlx")
  expect_equal(dim(as.matrix(perm_mat)), c(4L, 3L))

  # Each row of original should appear in permuted version
  original_rows <- as.matrix(as_mlx(mat))
  perm_rows <- as.matrix(perm_mat)

  # Check all values still present
  expect_equal(sort(as.vector(perm_rows)), 1:12)
})

test_that("mlx_rand_permutation permutes along specified axis", {
  set.seed(123)
  mat <- matrix(1:12, 4, 3)
  perm_cols <- mlx_rand_permutation(mat, axis = 1)

  expect_s3_class(perm_cols, "mlx")
  expect_equal(dim(as.matrix(perm_cols)), c(4L, 3L))

  # Check all values still present
  expect_equal(sort(as.vector(as.matrix(perm_cols))), 1:12)
})

test_that("mlx_key is deterministic for a given seed", {
  key1 <- mlx_key(123)
  key2 <- mlx_key(123)
  key3 <- mlx_key(124)

  expect_s3_class(key1, "mlx")
  expect_equal(key1$device, "cpu")
  expect_equal(as.matrix(key1), as.matrix(key2))
  expect_false(all(as.matrix(key1) == as.matrix(key3)))
})

test_that("mlx_key_split returns reproducible subkeys", {
  base_key <- mlx_key(321)
  splits <- mlx_key_split(base_key, num = 3)

  expect_length(splits, 3)
  expect_true(all(vapply(splits, inherits, logical(1), what = "mlx")))

  splits_again <- mlx_key_split(mlx_key(321), num = 3)
  for (i in seq_along(splits)) {
    expect_equal(as.matrix(splits[[i]]), as.matrix(splits_again[[i]]))
  }
  expect_false(all(as.matrix(splits[[1]]) == as.matrix(splits[[2]])))
})

test_that("mlx_key_bits produces deterministic bit patterns", {
  key <- mlx_key(777)
  bits1 <- mlx_key_bits(c(4L, 2L), key = key)
  bits2 <- mlx_key_bits(c(4L, 2L), key = key)

  expect_s3_class(bits1, "mlx")
  expect_equal(mlx_dtype(bits1), "uint32")
  expect_equal(dim(bits1), c(4L, 2L))
  expect_equal(as.matrix(bits1), as.matrix(bits2))

  bits_wide <- mlx_key_bits(c(2L, 2L), width = 2L)
  expect_equal(mlx_dtype(bits_wide), "uint16")
})
