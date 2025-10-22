skip_on_cran()

set.seed(123)

shape <- c(256L, 256L)
mx_rng <- mlx_rand_uniform(shape, min = -1, max = 1)

expect_s3_class(mx_rng, "mlx")
expect_equal(mx_rng$dim, shape)
vals <- as.matrix(mx_rng)
expect_true(all(vals >= -1 & vals <= 1))

# Normal distribution
mx_norm <- mlx_rand_normal(shape, mean = 2, sd = 0.5)
expect_s3_class(mx_norm, "mlx")
expect_equal(mx_norm$dim, shape)
norm_vals <- as.matrix(mx_norm)
expect_true(abs(mean(norm_vals) - 2) < 0.1)
expect_true(sd(norm_vals) > 0)

# Bernoulli distribution
mx_bern <- mlx_rand_bernoulli(shape, prob = 0.3)
expect_s3_class(mx_bern, "mlx")
expect_equal(mx_bern$dim, shape)
bern_vals <- as.matrix(mx_bern)
expect_true(all(bern_vals %in% c(0, 1)))

# Gumbel distribution
mx_gumbel <- mlx_rand_gumbel(shape)
expect_s3_class(mx_gumbel, "mlx")
expect_equal(mx_gumbel$dim, shape)
gumbel_vals <- as.matrix(mx_gumbel)
# Check that values are finite
expect_true(all(is.finite(gumbel_vals)))
# Gumbel distribution mean should be close to Euler-Mascheroni constant (~0.5772)
expect_true(abs(mean(gumbel_vals) - 0.5772) < 0.05)
# Standard deviation should be close to pi/sqrt(6) (~1.2825)
expect_true(abs(sd(gumbel_vals) - 1.2825) < 0.1)

# Truncated normal distribution
mx_tnorm <- mlx_rand_truncated_normal(-1, 1, c(100L, 100L))
expect_s3_class(mx_tnorm, "mlx")
expect_equal(mx_tnorm$dim, c(100L, 100L))
tnorm_vals <- as.matrix(mx_tnorm)
# Check all values are within bounds
expect_true(all(tnorm_vals >= -1 & tnorm_vals <= 1))
# Check that values are finite
expect_true(all(is.finite(tnorm_vals)))

# Test truncated normal with different bounds
mx_tnorm2 <- mlx_rand_truncated_normal(0, 10, c(50L, 50L))
tnorm2_vals <- as.matrix(mx_tnorm2)
expect_true(all(tnorm2_vals >= 0 & tnorm2_vals <= 10))

# Test truncated normal with CPU device
mx_tnorm_cpu <- mlx_rand_truncated_normal(-2, 2, c(40L, 40L), device = "cpu")
expect_equal(mx_tnorm_cpu$device, "cpu")
expect_equal(mx_tnorm_cpu$dim, c(40L, 40L))

# Multivariate normal distribution (requires CPU due to SVD limitation)
mvn_mean <- as_mlx(c(0, 0), device = "cpu")
mvn_cov <- as_mlx(matrix(c(1, 0, 0, 1), 2, 2), device = "cpu")
mx_mvn <- mlx_rand_multivariate_normal(c(10L, 2L), mvn_mean, mvn_cov, device = "cpu")
expect_s3_class(mx_mvn, "mlx")
expect_equal(mx_mvn$device, "cpu")
expect_equal(mx_mvn$dtype, "float32")
mvn_vals <- as.vector(as.matrix(mx_mvn))
expect_true(all(is.finite(mvn_vals)))

# Test multivariate normal with non-identity covariance
mvn_mean2 <- as_mlx(c(1, 2), device = "cpu")
mvn_cov2 <- as_mlx(matrix(c(2, 0.5, 0.5, 1), 2, 2), device = "cpu")
mx_mvn2 <- mlx_rand_multivariate_normal(c(5L, 2L), mvn_mean2, mvn_cov2, device = "cpu")
expect_s3_class(mx_mvn2, "mlx")
expect_equal(mx_mvn2$device, "cpu")
mvn2_vals <- as.vector(as.matrix(mx_mvn2))
expect_true(all(is.finite(mvn2_vals)))


