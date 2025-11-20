test_that("mlx_conv_transpose1d works", {
  # Simple 1D transposed convolution test
  input <- mlx_array(1:12, dim = c(1, 4, 3))  # batch=1, length=4, channels=3
  weight <- mlx_array(rep(1:6, 2), dim = c(2, 2, 3))  # out_channels=2, kernel=2, in_channels=3

  result <- mlx_conv_transpose1d(input, weight)

  expect_s3_class(result, "mlx")
  expect_equal(length(dim(result)), 3)
  expect_equal(dim(result)[1], 1)  # batch size preserved

  # With stride - upsamples the output
  result_stride <- mlx_conv_transpose1d(input, weight, stride = 2)
  expect_s3_class(result_stride, "mlx")
  expect_true(dim(result_stride)[2] > dim(result)[2])  # Larger output with stride > 1

  # With output padding
  result_opad <- mlx_conv_transpose1d(input, weight, output_padding = 1)
  expect_s3_class(result_opad, "mlx")
})

test_that("mlx_conv_transpose2d works", {
  set.seed(42)
  # Simple 2D transposed convolution test
  input <- mlx_array(rnorm(1*3*3*4), dim = c(1, 3, 3, 4))  # batch=1, 3x3, channels=4
  weight <- mlx_array(rnorm(8*3*3*4), dim = c(8, 3, 3, 4)) # out=8, 3x3 kernel, in=4

  result <- mlx_conv_transpose2d(input, weight)

  expect_s3_class(result, "mlx")
  expect_equal(length(dim(result)), 4)
  expect_equal(dim(result)[1], 1)  # batch size
  expect_equal(dim(result)[4], 8)  # output channels

  # With stride - upsamples spatial dimensions
  result_stride <- mlx_conv_transpose2d(input, weight, stride = c(2, 2))
  expect_s3_class(result_stride, "mlx")
  expect_true(dim(result_stride)[2] > dim(result)[2])  # Larger height with stride
  expect_true(dim(result_stride)[3] > dim(result)[3])  # Larger width with stride

  # With output padding
  result_opad <- mlx_conv_transpose2d(input, weight, output_padding = c(1, 1))
  expect_s3_class(result_opad, "mlx")
  expect_true(dim(result_opad)[2] > dim(result)[2])  # Slightly larger output

  # Scalar stride/padding/output_padding should work
  result_scalar <- mlx_conv_transpose2d(input, weight, stride = 1, padding = 1, output_padding = 0)
  expect_s3_class(result_scalar, "mlx")
})

test_that("mlx_conv_transpose3d works", {
  set.seed(42)
  # Simple 3D transposed convolution test
  input <- mlx_array(rnorm(1*2*2*2*3), dim = c(1, 2, 2, 2, 3))  # batch=1, 2x2x2, channels=3
  weight <- mlx_array(rnorm(4*2*2*2*3), dim = c(4, 2, 2, 2, 3)) # out=4, 2x2x2 kernel, in=3

  result <- mlx_conv_transpose3d(input, weight)

  expect_s3_class(result, "mlx")
  expect_equal(length(dim(result)), 5)
  expect_equal(dim(result)[1], 1)  # batch size
  expect_equal(dim(result)[5], 4)  # output channels

  # With stride - upsamples volumetric dimensions
  result_stride <- mlx_conv_transpose3d(input, weight, stride = c(2, 2, 2))
  expect_s3_class(result_stride, "mlx")
  expect_true(all(dim(result_stride)[2:4] > dim(result)[2:4]))  # All spatial dims larger

  # With output padding
  result_opad <- mlx_conv_transpose3d(input, weight, output_padding = c(1, 1, 1))
  expect_s3_class(result_opad, "mlx")

  # Scalar parameters should work
  result_scalar <- mlx_conv_transpose3d(input, weight, stride = 1, padding = 0,
                                        dilation = 1, output_padding = 0)
  expect_s3_class(result_scalar, "mlx")
})

test_that("transpose conv upsamples correctly", {
  set.seed(123)
  # Test that transpose conv upsamples as expected
  input_small <- mlx_array(rnorm(1*4*4*8), dim = c(1, 4, 4, 8))
  weight <- mlx_array(rnorm(16*3*3*8), dim = c(16, 3, 3, 8))

  # Apply transpose conv with stride 2 (upsamples)
  upsampled <- mlx_conv_transpose2d(input_small, weight, stride = c(2, 2), padding = c(1, 1))

  # Output spatial dims should be roughly 2x the input (approximately)
  # Formula: output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
  # With input=4, stride=2, padding=1, kernel=3: (4-1)*2 - 2*1 + 3 = 7
  expect_equal(dim(upsampled)[2], 7)
  expect_equal(dim(upsampled)[3], 7)
  expect_equal(dim(upsampled)[4], 16)  # output channels
})
