test_that("as_mlx converts R objects correctly", {
  skip_if_not_installed("Rmlx")
  skip_on_cran()

  # Vector
  v <- 1:10
  v_mlx <- as_mlx(v)
  expect_s3_class(v_mlx, "mlx")
  expect_equal(v_mlx$dim, 10L)

  # Matrix
  m <- matrix(1:12, 3, 4)
  m_mlx <- as_mlx(m)
  expect_s3_class(m_mlx, "mlx")
  expect_equal(m_mlx$dim, c(3L, 4L))
  expect_equal(m_mlx$dtype, "float64")
})

test_that("roundtrip conversion preserves values", {
  skip_if_not_installed("Rmlx")
  skip_on_cran()

  m <- matrix(rnorm(20), 4, 5)
  m_mlx <- as_mlx(m)
  m_back <- as.matrix(m_mlx)

  expect_equal(m_back, m, tolerance = 1e-6)
})

test_that("dtype argument works", {
  skip_if_not_installed("Rmlx")
  skip_on_cran()

  m <- matrix(1:12, 3, 4)
  m_fp32 <- as_mlx(m, dtype = "float32")
  m_fp64 <- as_mlx(m, dtype = "float64")

  expect_equal(m_fp32$dtype, "float32")
  expect_equal(m_fp64$dtype, "float64")
})

test_that("is.mlx works", {
  m <- matrix(1:12, 3, 4)
  m_mlx <- as_mlx(m)

  expect_true(is.mlx(m_mlx))
  expect_false(is.mlx(m))
  expect_false(is.mlx(NULL))
  expect_false(is.mlx(list()))
})

test_that("mlx_eval runs without error", {
  skip_if_not_installed("Rmlx")
  skip_on_cran()

  m <- matrix(1:12, 3, 4)
  m_mlx <- as_mlx(m)

  expect_invisible(mlx_eval(m_mlx))
})
