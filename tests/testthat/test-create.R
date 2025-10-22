test_that("mlx_zeros creates tensors of zeros", {
  tens <- mlx_zeros(c(2, 3), dtype = "float32", device = "cpu")
  expect_s3_class(tens, "mlx")
  expect_equal(mlx_dim(tens), c(2L, 3L))
  expect_equal(mlx_dtype(tens), "float32")
  expect_equal(as.matrix(tens), matrix(0, 2, 3), tolerance = 1e-6)
})

test_that("mlx_ones creates tensors of ones", {
  tens <- mlx_ones(c(2, 2), dtype = "float32", device = "cpu")
  expect_equal(as.matrix(tens), matrix(1, 2, 2), tolerance = 1e-6)
  expect_equal(mlx_dtype(tens), "float32")
})

test_that("mlx_full fills with scalars", {
  filled <- mlx_full(c(2, 2), 3)
  expect_equal(as.matrix(filled), matrix(3, 2, 2), tolerance = 1e-6)

  complex_filled <- mlx_full(c(2, 1), 1 + 2i, dtype = "complex64")
  expect_equal(as.matrix(complex_filled), matrix(1 + 2i, 2, 1))
  expect_equal(mlx_dtype(complex_filled), "complex64")

  bool_filled <- mlx_full(c(2, 2), TRUE, dtype = "bool")
  expect_equal(as.matrix(bool_filled), matrix(TRUE, 2, 2))
  expect_equal(mlx_dtype(bool_filled), "bool")
})

test_that("identity helpers work", {
  eye <- mlx_eye(3, k = 1)
  expected_eye <- matrix(0, 3, 3)
  expected_eye[cbind(1:2, 2:3)] <- 1
  expect_equal(as.matrix(eye), expected_eye, tolerance = 1e-6)

  ident <- mlx_identity(4)
  expect_equal(as.matrix(ident), diag(4), tolerance = 1e-6)
})

test_that("range helpers produce expected sequences", {
  ar <- mlx_arange(5)
  expect_equal(as.vector(as.matrix(ar)), 0:4, tolerance = 1e-6)

  ar_custom <- mlx_arange(5, start = 1, step = 2)
  expect_equal(as.vector(as.matrix(ar_custom)), c(1, 3), tolerance = 1e-6)

  lin <- mlx_linspace(0, 1, num = 5)
  expect_equal(as.vector(as.matrix(lin)), seq(0, 1, length.out = 5), tolerance = 1e-6)
})
