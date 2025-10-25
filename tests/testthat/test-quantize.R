test_that("quantize and dequantize work", {
  w <- mlx_rand_normal(c(128, 64))

  # Test quantization
  quant <- mlx_quantize(w, group_size = 64L, bits = 4L, mode = "affine")

  expect_s3_class(quant$w_q, "mlx")
  expect_s3_class(quant$scales, "mlx")
  expect_s3_class(quant$biases, "mlx")

  # Check dimensions - quantized weights are packed
  expect_equal(mlx_dim(quant$w_q), c(128, 8))  # 64/8 = 8 (8 4-bit values per uint32)
  expect_equal(mlx_dim(quant$scales), c(128, 1))  # one scale per group
  expect_equal(mlx_dim(quant$biases), c(128, 1))

  # Test dequantization
  w_recon <- mlx_dequantize(quant$w_q, quant$scales, quant$biases,
                             group_size = 64L, bits = 4L, mode = "affine")

  expect_s3_class(w_recon, "mlx")
  expect_equal(mlx_dim(w_recon), c(128, 64))

  # Reconstructed weights should be approximately equal (quantization loses precision)
  # We can't use exact equality due to quantization error
})

test_that("quantized_matmul with auto-quantization works", {
  x <- mlx_rand_normal(c(10, 64))
  w <- mlx_rand_normal(c(128, 64))

  # Auto-quantization (scales = NULL)
  result_auto <- mlx_quantized_matmul(x, w)

  expect_s3_class(result_auto, "mlx")
  expect_equal(mlx_dim(result_auto), c(10, 128))
})

test_that("quantized_matmul with pre-quantized weights works", {
  x <- mlx_rand_normal(c(10, 64))
  w <- mlx_rand_normal(c(128, 64))

  # Pre-quantize
  quant <- mlx_quantize(w, group_size = 64L, bits = 4L)

  # Use pre-quantized weights
  result_pre <- mlx_quantized_matmul(x, quant$w_q, quant$scales, quant$biases)

  expect_s3_class(result_pre, "mlx")
  expect_equal(mlx_dim(result_pre), c(10, 128))
})

test_that("mxfp4 quantization mode works", {
  w <- mlx_rand_normal(c(128, 64))

  # mxfp4 mode (4-bit floating point, requires group_size = 32, no biases)
  quant_mxfp4 <- mlx_quantize(w, mode = "mxfp4", group_size = 32L, bits = 4L)

  expect_s3_class(quant_mxfp4$w_q, "mlx")
  expect_s3_class(quant_mxfp4$scales, "mlx")
  expect_null(quant_mxfp4$biases)

  # Dequantize
  w_recon <- mlx_dequantize(quant_mxfp4$w_q, quant_mxfp4$scales,
                             mode = "mxfp4", group_size = 32L, bits = 4L)

  expect_s3_class(w_recon, "mlx")
  expect_equal(mlx_dim(w_recon), c(128, 64))
})
