test_that("quantize and dequantize work", {
  w <- mlx_rand_normal(c(128, 64))

  # Test quantization
  quant <- mlx_quantize(w, group_size = 64L, bits = 4L, mode = "affine")

  expect_s3_class(quant$w_q, "mlx")
  expect_s3_class(quant$scales, "mlx")
  expect_s3_class(quant$biases, "mlx")

  # Check dimensions - quantized weights are packed
  expect_equal(mlx_shape(quant$w_q), c(128, 8))  # 64/8 = 8 (8 4-bit values per uint32)
  expect_equal(mlx_shape(quant$scales), c(128, 1))  # one scale per group
  expect_equal(mlx_shape(quant$biases), c(128, 1))

  # Test dequantization
  w_recon <- mlx_dequantize(quant$w_q, quant$scales, quant$biases,
                             group_size = 64L, bits = 4L, mode = "affine")

  expect_s3_class(w_recon, "mlx")
  expect_equal(mlx_shape(w_recon), c(128, 64))

  # Reconstructed weights should be approximately equal (quantization loses precision)
  # We can't use exact equality due to quantization error
})

test_that("quantized_matmul with auto-quantization works", {
  x <- mlx_rand_normal(c(10, 64))
  w <- mlx_rand_normal(c(128, 64))

  # Auto-quantization (scales = NULL)
  result_auto <- mlx_quantized_matmul(x, w)

  expect_s3_class(result_auto, "mlx")
  expect_equal(mlx_shape(result_auto), c(10, 128))
})

test_that("quantized_matmul with pre-quantized weights works", {
  x <- mlx_rand_normal(c(10, 64))
  w <- mlx_rand_normal(c(128, 64))

  # Pre-quantize
  quant <- mlx_quantize(w, group_size = 64L, bits = 4L)

  # Use pre-quantized weights
  result_pre <- mlx_quantized_matmul(x, quant$w_q, quant$scales, quant$biases)

  expect_s3_class(result_pre, "mlx")
  expect_equal(mlx_shape(result_pre), c(10, 128))
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
  expect_equal(mlx_shape(w_recon), c(128, 64))
})

test_that("gather_qmm matches quantized_matmul without gathers", {
  seed <- as.integer(format(Sys.Date(), "%Y%m%d"))
  set.seed(seed)

  x <- mlx_matrix(rnorm(64), nrow = 2, ncol = 32, dtype = "float32", device = "cpu")
  w <- mlx_matrix(rnorm(96), nrow = 3, ncol = 32, dtype = "float32", device = "cpu")

  quant <- mlx_quantize(w, group_size = 32L, bits = 4L, mode = "affine")

  mm_ref <- mlx_quantized_matmul(
    x, quant$w_q, quant$scales, quant$biases,
    group_size = 32L, bits = 4L, mode = "affine", transpose = TRUE, device = "cpu"
  )
  mm_gather <- mlx_gather_qmm(
    x, quant$w_q, quant$scales, quant$biases,
    group_size = 32L, bits = 4L, mode = "affine", transpose = TRUE, device = "cpu"
  )

  expect_equal(as_r(mm_gather), as_r(mm_ref), tolerance = 1e-4)
})

test_that("gather_qmm applies lhs/rhs indices correctly", {
  seed <- as.integer(format(Sys.Date(), "%Y%m%d"))
  set.seed(seed)

  # Batch 3, M=2, K=32
  x_data <- array(rnorm(3 * 2 * 32), dim = c(3, 2, 32))
  x <- mlx_array(x_data, dim = c(3, 2, 32), dtype = "float32", device = "cpu")

  # Batch 3, N=5, K=32
  w_data <- array(rnorm(3 * 5 * 32), dim = c(3, 5, 32))
  w <- mlx_array(w_data, dim = c(3, 5, 32), dtype = "float32", device = "cpu")

  quant <- mlx_quantize(w, group_size = 32L, bits = 4L, mode = "affine")
  w_dequant <- mlx_dequantize(quant$w_q, quant$scales, quant$biases,
                              group_size = 32L, bits = 4L, mode = "affine")

  lhs_idx_r <- c(1L, 3L)   # 1-based for R arrays
  rhs_idx_r <- c(3L, 2L)

  res <- mlx_gather_qmm(
    x, quant$w_q, quant$scales, quant$biases,
    lhs_indices = lhs_idx_r, rhs_indices = rhs_idx_r,
    group_size = 32L, bits = 4L, mode = "affine", transpose = TRUE, sorted_indices = FALSE,
    device = "cpu"
  )

  x_ref <- x_data[lhs_idx_r, , , drop = FALSE]
  w_ref <- as.array(w_dequant)[rhs_idx_r, , , drop = FALSE]

  ref_mat <- vapply(
    seq_along(lhs_idx_r),
    function(b) x_ref[b, , ] %*% t(w_ref[b, , ]),
    FUN.VALUE = matrix(0, nrow = 2, ncol = 5)
  )
  dim(ref_mat) <- c(2, 5, length(lhs_idx_r))
  ref_mat <- aperm(ref_mat, c(3, 1, 2)) # batch, M, N

  expect_equal(as.array(res), ref_mat, tolerance = 1e-3)
})
