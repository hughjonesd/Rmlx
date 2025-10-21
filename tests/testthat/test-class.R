test_that("as_mlx converts R objects correctly", {
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
  expect_equal(m_mlx$dtype, "float32")  # Default is float32 for GPU compatibility
})

test_that("roundtrip conversion preserves values", {
  m <- matrix(rnorm(20), 4, 5)
  m_mlx <- as_mlx(m)
  m_back <- as.matrix(m_mlx)

  expect_equal(m_back, m, tolerance = 1e-6)
})

test_that("roundtrip conversion preserves higher-dimensional arrays", {
  arr <- array(seq_len(24), dim = c(2, 3, 4))
  arr_mlx <- as_mlx(arr)
  arr_back <- as.array(arr_mlx)

  expect_equal(arr_back, arr, tolerance = 1e-6)
})

test_that("dtype argument works", {
  m <- matrix(1:12, 3, 4)
  m_fp32 <- as_mlx(m, dtype = "float32")
  expect_warning(as_mlx(m, dtype = "float64"), "stored in float32", fixed = TRUE)

  expect_equal(m_fp32$dtype, "float32")
})

test_that("logical inputs create boolean MLX arrays", {
  m <- matrix(c(TRUE, FALSE, TRUE, TRUE), 2, 2)
  m_mlx <- as_mlx(m)

  expect_s3_class(m_mlx, "mlx")
  expect_equal(m_mlx$dtype, "bool")
  expect_equal(m_mlx$dim, dim(m))
  expect_identical(as.matrix(m_mlx), m)
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
  m <- matrix(1:12, 3, 4)
  m_mlx <- as_mlx(m)

  expect_invisible(mlx_eval(m_mlx))
})

test_that("as.vector.mlx works for 1D arrays", {
  v <- 1:10
  v_mlx <- as_mlx(v)

  v_back <- as.vector(v_mlx)
  expect_equal(v_back, as.numeric(v))
})

test_that("as.vector.mlx fails for multi-dimensional arrays", {
  m <- matrix(1:12, 3, 4)
  m_mlx <- as_mlx(m)

  expect_error(
    as.vector(m_mlx),
    "Cannot convert multi-dimensional mlx array to vector"
  )
})
