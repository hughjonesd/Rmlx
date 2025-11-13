test_that("as_mlx converts R objects correctly", {
  # Vector
  v <- 1:10
  v_mlx <- as_mlx(v)
  expect_s3_class(v_mlx, "mlx")
  expect_equal(dim(v_mlx), 10L)

  # Matrix
  m <- matrix(1:12, 3, 4)
  m_mlx <- as_mlx(m)
  expect_s3_class(m_mlx, "mlx")
  expect_equal(dim(m_mlx), c(3L, 4L))
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
  expect_equal(dim(m_mlx), dim(m))
  expect_identical(as.matrix(m_mlx), m)
})

test_that("complex inputs create complex MLX arrays", {
  m <- matrix(complex(real = 1:4, imaginary = seq(0.1, 0.4, by = 0.1)), 2, 2)
  m_mlx <- as_mlx(m)

  expect_s3_class(m_mlx, "mlx")
  expect_equal(m_mlx$dtype, "complex64")
  expect_equal(dim(m_mlx), dim(m))
  expect_equal(as.matrix(m_mlx), m, tolerance = 1e-5)
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

test_that("as.vector.mlx flattens multi-dimensional arrays", {
  m <- matrix(1:12, 3, 4)
  m_mlx <- as_mlx(m)

  result <- as.vector(m_mlx)

  # Should flatten in column-major order (R's default)
  expect_equal(result, as.vector(m))
})

test_that("as.numeric.mlx works for different dtypes", {
  # Float (already numeric)
  x_float <- as_mlx(c(1.5, 2.5, 3.5), dtype = "float32")
  result_float <- as.numeric(x_float)
  expect_type(result_float, "double")
  expect_equal(result_float, c(1.5, 2.5, 3.5))

  # Integer (converts to numeric)
  x_int <- as_mlx(c(1L, 2L, 3L), dtype = "int32")
  result_int <- as.numeric(x_int)
  expect_type(result_int, "double")
  expect_equal(result_int, c(1, 2, 3))

  # Boolean (converts to 0/1)
  x_bool <- as_mlx(c(TRUE, FALSE, TRUE))
  result_bool <- as.numeric(x_bool)
  expect_type(result_bool, "double")
  expect_equal(result_bool, c(1, 0, 1))
})

test_that("as.numeric.mlx drops dimensions", {
  # Matrix
  m <- matrix(1:6, 2, 3)
  m_mlx <- as_mlx(m)

  result <- as.numeric(m_mlx)

  expect_type(result, "double")
  expect_equal(result, as.numeric(as.vector(m)))

  # 3D array
  arr <- array(1:24, dim = c(2, 3, 4))
  arr_mlx <- as_mlx(arr)

  result_3d <- as.numeric(arr_mlx)

  expect_type(result_3d, "double")
  expect_equal(result_3d, as.numeric(as.vector(arr)))
})

test_that("row()/col() match base results for mlx matrices", {
  mat <- matrix(seq_len(12), 3, 4)
  mat_mlx <- as_mlx(mat)

  expect_equal(as.matrix(row(mat_mlx)), base::row(mat))
  expect_equal(as.matrix(col(mat_mlx)), base::col(mat))
})

test_that("asplit() returns mlx slices matching base arrays", {
  x <- matrix(1:6, 2, 3)
  x_mlx <- as_mlx(x)

  base_split <- base::asplit(x, 1)
  mlx_split <- asplit(x_mlx, 1)

  expect_true(all(vapply(mlx_split, is.mlx, logical(1))))
  expect_equal(lapply(mlx_split, as.vector), lapply(base_split, as.vector))

  mlx_split_drop <- asplit(x_mlx, 1, drop = TRUE)
  base_split_drop <- base::asplit(x, 1, drop = TRUE)
  expect_equal(lapply(mlx_split_drop, as.vector), lapply(base_split_drop, as.vector))
})

test_that("backsolve() delegates to mlx_solve_triangular", {
  r <- matrix(c(3, 1, 0, 2), 2, 2)
  b_mat <- matrix(c(5, 4), 2, 1)
  b_vec <- c(5, 4)

  expected_mat <- base::backsolve(r, b_mat)
  expected_vec <- base::backsolve(r, b_vec)

  res_mat <- backsolve(as_mlx(r), as_mlx(b_mat), upper.tri = TRUE)
  res_vec <- backsolve(as_mlx(r), as_mlx(b_vec), upper.tri = TRUE)

  expect_s3_class(res_mat, "mlx")
  expect_equal(as.matrix(res_mat), expected_mat, tolerance = 1e-6)
  expect_equal(as.vector(res_vec), as.vector(expected_vec), tolerance = 1e-6)
})

test_that("scale.mlx matches base scale", {
  set.seed(123)
  mat <- matrix(rnorm(12), 3, 4)
  mlx_res <- scale(as_mlx(mat))
  base_res <- scale(mat)
  expect_equal(as.matrix(mlx_res), unclass(base_res), tolerance = 1e-6, ignore_attr = TRUE)
  expect_s3_class(attr(mlx_res, "scaled:center"), "mlx")
  expect_s3_class(attr(mlx_res, "scaled:scale"), "mlx")
  expect_equal(as.vector(as.matrix(attr(mlx_res, "scaled:center"))), attr(base_res, "scaled:center"), tolerance = 1e-6)
  expect_equal(as.vector(as.matrix(attr(mlx_res, "scaled:scale"))), attr(base_res, "scaled:scale"), tolerance = 1e-6)

  mlx_res2 <- scale(as_mlx(mat), center = FALSE, scale = c(1, 2, 3, 4))
  base_res2 <- scale(mat, center = FALSE, scale = c(1, 2, 3, 4))
  expect_equal(as.matrix(mlx_res2), unclass(base_res2), tolerance = 1e-6, ignore_attr = TRUE)
  expect_null(attr(mlx_res2, "scaled:center"))
  expect_equal(attr(mlx_res2, "scaled:scale"), c(1, 2, 3, 4))
})
