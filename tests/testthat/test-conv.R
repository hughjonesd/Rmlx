test_that("mlx_conv1d works", {
  # Simple 1D convolution test
  input <- mlx_array(1:12, dim = c(1, 4, 3))  # batch=1, length=4, channels=3
  weight <- mlx_array(rep(1:6, 2), dim = c(2, 2, 3))  # out_channels=2, kernel=2, in_channels=3

  result <- mlx_conv1d(input, weight)

  expect_s3_class(result, "mlx")
  expect_equal(length(dim(result)), 3)

  # With stride and padding
  result_stride <- mlx_conv1d(input, weight, stride = 2)
  expect_s3_class(result_stride, "mlx")

  result_pad <- mlx_conv1d(input, weight, padding = 1)
  expect_s3_class(result_pad, "mlx")
})

test_that("mlx_conv2d works", {
  set.seed(42)
  # Simple 2D convolution test
  input <- mlx_array(rnorm(1*5*5*3), dim = c(1, 5, 5, 3))  # batch=1, 5x5, channels=3
  weight <- mlx_array(rnorm(8*3*3*3), dim = c(8, 3, 3, 3))  # out=8, 3x3 kernel, in=3

  result <- mlx_conv2d(input, weight)

  expect_s3_class(result, "mlx")
  expect_equal(length(dim(result)), 4)
  expect_equal(dim(result)[1], 1)  # batch size
  expect_equal(dim(result)[4], 8)  # output channels

  # With padding (same padding)
  result_pad <- mlx_conv2d(input, weight, padding = c(1, 1))
  expect_equal(dim(result_pad)[2:3], c(5, 5))  # Same spatial dims with padding

  # With stride
  result_stride <- mlx_conv2d(input, weight, stride = c(2, 2))
  expect_s3_class(result_stride, "mlx")
  expect_true(all(dim(result_stride)[2:3] < c(5, 5)))  # Smaller output with stride

  # Scalar stride/padding should work
  result_scalar <- mlx_conv2d(input, weight, stride = 1, padding = 1)
  expect_s3_class(result_scalar, "mlx")
})

test_that("mlx_conv3d works", {
  set.seed(42)
  # Simple 3D convolution test
  input <- mlx_array(rnorm(1*3*3*3*2), dim = c(1, 3, 3, 3, 2))  # batch=1, 3x3x3, channels=2
  weight <- mlx_array(rnorm(4*2*2*2*2), dim = c(4, 2, 2, 2, 2))  # out=4, 2x2x2 kernel, in=2

  result <- mlx_conv3d(input, weight)

  expect_s3_class(result, "mlx")
  expect_equal(length(dim(result)), 5)
  expect_equal(dim(result)[1], 1)  # batch size
  expect_equal(dim(result)[5], 4)  # output channels

  # With padding
  result_pad <- mlx_conv3d(input, weight, padding = c(1, 1, 1))
  expect_s3_class(result_pad, "mlx")

  # Scalar parameters should work
  result_scalar <- mlx_conv3d(input, weight, stride = 1, padding = 0, dilation = 1)
  expect_s3_class(result_scalar, "mlx")
})
