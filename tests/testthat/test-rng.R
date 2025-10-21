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

