test_that("float64 arrays roundtrip on CPU", {
  with_device("cpu", {
    values <- c(1, 2 + 2^-40, pi)
    vec <- as_mlx(values, dtype = "float64")
    mat_data <- matrix(values[rep(1:3, 2)], 2, 3)
    mat <- as_mlx(mat_data, dtype = "float64")
    arr_data <- array(seq(1, 8) + 2^-40, dim = c(2, 2, 2))
    arr <- as_mlx(arr_data, dtype = "float64")

    expect_equal(mlx_dtype(vec), "float64")
    expect_equal(as.vector(vec), values, tolerance = 1e-12)

    expect_equal(mlx_dtype(mat), "float64")
    expect_equal(as.matrix(mat), mat_data, tolerance = 1e-12)

    expect_equal(mlx_dtype(arr), "float64")
    expect_equal(as.array(arr), arr_data, tolerance = 1e-12)
  })
})

test_that("constructors create true float64 on CPU", {
  with_device("cpu", {
    constructors <- list(
      mlx_array(1:4, dim = c(2, 2), dtype = "float64"),
      mlx_vector(c(1, 2), dtype = "float64"),
      mlx_matrix(1:4, nrow = 2, dtype = "float64"),
      mlx_scalar(1, dtype = "float64"),
      mlx_zeros(c(2, 2), dtype = "float64"),
      mlx_ones(c(2, 2), dtype = "float64"),
      mlx_full(c(2, 2), 1, dtype = "float64"),
      mlx_eye(2, dtype = "float64"),
      mlx_identity(2, dtype = "float64"),
      mlx_tri(2, dtype = "float64"),
      mlx_arange(1, 3, dtype = "float64"),
      mlx_linspace(0, 1, dtype = "float64")
    )

    expect_true(all(vapply(constructors, mlx_dtype, character(1)) == "float64"))

    x <- mlx_vector(1:3, dtype = "float32")
    cast <- mlx_cast(x, dtype = "float64")
    expect_equal(mlx_dtype(cast), "float64")

    normal <- mlx_rand_normal(c(2, 2), dtype = "float64")
    uniform <- mlx_rand_uniform(c(2, 2), dtype = "float64")
    expect_equal(mlx_dtype(normal), "float64")
    expect_equal(mlx_dtype(uniform), "float64")
  })
})

test_that("float64 arithmetic and linear algebra stay on CPU", {
  with_device("cpu", {
    x <- as_mlx(matrix(c(1, 2, 3, 4), 2, 2), dtype = "float64")
    y <- as_mlx(matrix(c(5, 6, 7, 8), 2, 2), dtype = "float64")

    sum <- x + 1
    expect_equal(mlx_dtype(sum), "float64")
    expect_equal(as.matrix(sum), matrix(c(2, 3, 4, 5), 2, 2), tolerance = 1e-12)

    prod <- x %*% y
    expect_equal(mlx_dtype(prod), "float64")
    expect_equal(as.matrix(prod), as.matrix(x) %*% as.matrix(y), tolerance = 1e-12)

    expect_equal(mlx_dtype(mlx_sum(x)), "float64")
    expect_equal(as.vector(mlx_sum(x)), sum(as.matrix(x)), tolerance = 1e-12)

    a <- as_mlx(matrix(c(3, 1, 1, 2), 2, 2), dtype = "float64")
    b <- as_mlx(c(9, 8), dtype = "float64")
    sol <- solve(a, b, device = "cpu")
    expect_equal(mlx_dtype(sol), "float64")
    expect_equal(as.vector(sol), solve(as.matrix(a), as.vector(b)), tolerance = 1e-10)
  })
})


test_that("float64 errors when the current device is GPU", {
  skip_if_not(mlx_has_gpu())
  x <- with_device("cpu", as_mlx(1:3, dtype = "float64"))

  with_device("gpu", {
    expect_error(x + 1, "float64", fixed = TRUE)
    expect_error(mlx_stack(x, x), "float64", fixed = TRUE)
    expect_error(mlx_where(x > 1, x, x), "float64", fixed = TRUE)
  })
})

test_that("GPU float32 can be explicitly finished on CPU float64", {
  skip_if_not(mlx_has_gpu())
  y <- with_device("gpu", {
    x <- as_mlx(1:3, dtype = "float32")
    x + 1
  })
  out <- with_device("cpu", {
    z <- mlx_cast(y, dtype = "float64")
    z + 0.25
  })

  expect_equal(mlx_dtype(out), "float64")
  expect_equal(with_device("cpu", as.vector(out)), c(2.25, 3.25, 4.25), tolerance = 1e-12)
})
