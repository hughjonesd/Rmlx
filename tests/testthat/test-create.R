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
  expect_equal(as.vector(ar), 0:4, tolerance = 1e-6)

  ar_custom <- mlx_arange(5, start = 1, step = 2)
  expect_equal(as.vector(ar_custom), c(1, 3), tolerance = 1e-6)

  lin <- mlx_linspace(0, 1, num = 5)
  expect_equal(as.vector(lin), seq(0, 1, length.out = 5), tolerance = 1e-6)
})

test_that("mlx_zeros_like matches source metadata", {
  base <- mlx_full(c(2, 3), 4, dtype = "float32", device = "cpu")
  zeros <- mlx_zeros_like(base)

  expect_s3_class(zeros, "mlx")
  expect_equal(mlx_dim(zeros), mlx_dim(base))
  expect_equal(mlx_dtype(zeros), mlx_dtype(base))
  expect_equal(zeros$device, base$device)
  expect_equal(as.matrix(zeros), matrix(0, 2, 3), tolerance = 1e-6)
})

test_that("mlx_zeros_like allows overriding dtype and device", {
  base <- mlx_full(c(2, 2), 1, dtype = "float32", device = mlx_default_device())
  zeros <- mlx_zeros_like(base, dtype = "int32", device = "cpu")

  expect_equal(mlx_dtype(zeros), "int32")
  expect_equal(zeros$device, "cpu")
  expect_equal(as.matrix(zeros), matrix(0, 2, 2))
})

test_that("mlx_ones_like mirrors shape and supports overrides", {
  base <- as_mlx(array(7L, dim = c(3, 1, 2)), dtype = "int16", device = "cpu")
  ones <- mlx_ones_like(base)

  expect_equal(mlx_dim(ones), mlx_dim(base))
  expect_equal(mlx_dtype(ones), "int16")
  expect_equal(ones$device, "cpu")
  expect_equal(as.array(ones), array(1, dim = c(3, 1, 2)))

  ones_float <- mlx_ones_like(base, dtype = "float32")
  expect_equal(mlx_dtype(ones_float), "float32")
})
