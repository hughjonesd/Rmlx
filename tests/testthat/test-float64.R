test_that("float64 arrays roundtrip on CPU", {
  values <- c(1, 2 + 2^-40, pi)
  vec <- as_mlx(values, dtype = "float64", device = "cpu")
  mat_data <- matrix(values[rep(1:3, 2)], 2, 3)
  mat <- as_mlx(mat_data, dtype = "float64", device = "cpu")
  arr_data <- array(seq(1, 8) + 2^-40, dim = c(2, 2, 2))
  arr <- as_mlx(arr_data, dtype = "float64", device = "cpu")

  expect_equal(mlx_dtype(vec), "float64")
  expect_equal(mlx_device(vec), "cpu")
  expect_equal(as.vector(vec), values, tolerance = 1e-12)

  expect_equal(mlx_dtype(mat), "float64")
  expect_equal(as.matrix(mat), mat_data, tolerance = 1e-12)

  expect_equal(mlx_dtype(arr), "float64")
  expect_equal(as.array(arr), arr_data, tolerance = 1e-12)
})

test_that("constructors create true float64 on CPU", {
  constructors <- list(
    mlx_array(1:4, dim = c(2, 2), dtype = "float64", device = "cpu"),
    mlx_vector(c(1, 2), dtype = "float64", device = "cpu"),
    mlx_matrix(1:4, nrow = 2, dtype = "float64", device = "cpu"),
    mlx_scalar(1, dtype = "float64", device = "cpu"),
    mlx_zeros(c(2, 2), dtype = "float64", device = "cpu"),
    mlx_ones(c(2, 2), dtype = "float64", device = "cpu"),
    mlx_full(c(2, 2), 1, dtype = "float64", device = "cpu"),
    mlx_eye(2, dtype = "float64", device = "cpu"),
    mlx_identity(2, dtype = "float64", device = "cpu"),
    mlx_tri(2, dtype = "float64", device = "cpu"),
    mlx_arange(1, 3, dtype = "float64", device = "cpu"),
    mlx_linspace(0, 1, dtype = "float64", device = "cpu")
  )

  expect_true(all(vapply(constructors, mlx_dtype, character(1)) == "float64"))
  expect_true(all(vapply(constructors, mlx_device, character(1)) == "cpu"))

  x <- mlx_vector(1:3, dtype = "float32", device = "cpu")
  cast <- mlx_cast(x, dtype = "float64", device = "cpu")
  expect_equal(mlx_dtype(cast), "float64")
  expect_equal(mlx_device(cast), "cpu")

  normal <- mlx_rand_normal(c(2, 2), dtype = "float64", device = "cpu")
  uniform <- mlx_rand_uniform(c(2, 2), dtype = "float64", device = "cpu")
  expect_equal(mlx_dtype(normal), "float64")
  expect_equal(mlx_dtype(uniform), "float64")
})

test_that("float64 arithmetic and linear algebra stay on CPU", {
  x <- as_mlx(matrix(c(1, 2, 3, 4), 2, 2), dtype = "float64", device = "cpu")
  y <- as_mlx(matrix(c(5, 6, 7, 8), 2, 2), dtype = "float64", device = "cpu")

  sum <- x + 1
  expect_equal(mlx_dtype(sum), "float64")
  expect_equal(mlx_device(sum), "cpu")
  expect_equal(as.matrix(sum), matrix(c(2, 3, 4, 5), 2, 2), tolerance = 1e-12)

  prod <- x %*% y
  expect_equal(mlx_dtype(prod), "float64")
  expect_equal(mlx_device(prod), "cpu")
  expect_equal(as.matrix(prod), as.matrix(x) %*% as.matrix(y), tolerance = 1e-12)

  expect_equal(mlx_dtype(mlx_sum(x)), "float64")
  expect_equal(as.vector(mlx_sum(x)), sum(as.matrix(x)), tolerance = 1e-12)

  a <- as_mlx(matrix(c(3, 1, 1, 2), 2, 2), dtype = "float64", device = "cpu")
  b <- as_mlx(c(9, 8), dtype = "float64", device = "cpu")
  sol <- solve(a, b)
  expect_equal(mlx_dtype(sol), "float64")
  expect_equal(mlx_device(sol), "cpu")
  expect_equal(as.vector(sol), solve(as.matrix(a), as.vector(b)), tolerance = 1e-10)
})

test_that("float64 cannot be created or moved to GPU", {
  expect_error(
    as_mlx(1:3, dtype = "float64", device = "gpu"),
    "float64 arrays are CPU-only",
    fixed = TRUE
  )
  expect_error(
    mlx_zeros(c(2, 2), dtype = "float64", device = "gpu"),
    "float64 arrays are CPU-only",
    fixed = TRUE
  )

  x <- as_mlx(1:3, dtype = "float64", device = "cpu")
  expect_error(
    mlx_cast(x, dtype = "float64", device = "gpu"),
    "float64 arrays are CPU-only",
    fixed = TRUE
  )
})

test_that("default GPU does not silently create float64 on CPU", {
  skip_if_not(mlx_has_gpu())
  old_device <- mlx_default_device()
  on.exit(mlx_default_device(old_device), add = TRUE)
  mlx_default_device("gpu")

  expect_error(
    as_mlx(1:3, dtype = "float64"),
    "float64 arrays are CPU-only",
    fixed = TRUE
  )
})

test_that("mixed CPU float64 and GPU operands error clearly", {
  skip_if_not(mlx_has_gpu())
  x <- as_mlx(1:3, dtype = "float64", device = "cpu")
  y <- as_mlx(1:3, dtype = "float32", device = "gpu")

  expect_error(x + y, "float64 arrays are CPU-only", fixed = TRUE)
  expect_error(mlx_stack(x, y), "float64 arrays are CPU-only", fixed = TRUE)
  expect_error(mlx_where(x > 1, x, y), "float64 arrays are CPU-only", fixed = TRUE)
})

test_that("GPU float32 can be explicitly finished on CPU float64", {
  skip_if_not(mlx_has_gpu())
  x <- as_mlx(1:3, dtype = "float32", device = "gpu")
  y <- x + 1
  z <- mlx_cast(y, dtype = "float64", device = "cpu")
  out <- z + 0.25

  expect_equal(mlx_dtype(out), "float64")
  expect_equal(mlx_device(out), "cpu")
  expect_equal(as.vector(out), c(2.25, 3.25, 4.25), tolerance = 1e-12)
})
