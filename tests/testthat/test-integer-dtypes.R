test_that("int32 dtype creation requires explicit dtype argument", {
  # R integers should convert to float32 by default (not int32)
  x_r <- 1:10
  x <- as_mlx(x_r)
  expect_equal(mlx_dtype(x), "float32")

  # Explicit dtype="int32" should create int32 array
  x_int <- as_mlx(x_r, dtype = "int32")
  expect_equal(mlx_dtype(x_int), "int32")

  # Matrix of integers
  m_r <- matrix(1:12, 3, 4)
  m <- as_mlx(m_r, dtype = "int32")
  expect_equal(mlx_dtype(m), "int32")
  expect_equal(mlx_dim(m), c(3, 4))
})

test_that("int64 dtype creation works", {
  x_r <- c(1, 2, 3, 4, 5)
  x <- as_mlx(x_r, dtype = "int64")
  expect_equal(mlx_dtype(x), "int64")

  # Large values that need int64
  large_vals <- c(2^31, 2^31 + 1)
  x_large <- as_mlx(large_vals, dtype = "int64")
  expect_equal(mlx_dtype(x_large), "int64")
})

test_that("int16 dtype creation works", {
  x_r <- c(1, 2, 3, 4, 5)
  x <- as_mlx(x_r, dtype = "int16")
  expect_equal(mlx_dtype(x), "int16")

  # Values in int16 range
  vals <- c(-32768, 0, 32767)
  x_range <- as_mlx(vals, dtype = "int16")
  expect_equal(mlx_dtype(x_range), "int16")
})

test_that("int8 dtype creation works", {
  x_r <- c(-128, 0, 127)
  x <- as_mlx(x_r, dtype = "int8")
  expect_equal(mlx_dtype(x), "int8")

  # Matrix
  m_r <- matrix(c(-10, 0, 10, 20), 2, 2)
  m <- as_mlx(m_r, dtype = "int8")
  expect_equal(mlx_dtype(m), "int8")
})

test_that("uint8 dtype creation works", {
  x_r <- c(0, 128, 255)
  x <- as_mlx(x_r, dtype = "uint8")
  expect_equal(mlx_dtype(x), "uint8")

  # Should handle conversion from positive integers
  pos_vals <- c(10, 20, 30)
  x_pos <- as_mlx(pos_vals, dtype = "uint8")
  expect_equal(mlx_dtype(x_pos), "uint8")
})

test_that("uint16 dtype creation works", {
  x_r <- c(0, 1000, 65535)
  x <- as_mlx(x_r, dtype = "uint16")
  expect_equal(mlx_dtype(x), "uint16")
})

test_that("uint32 dtype creation works", {
  x_r <- c(0, 1000, 2^20)
  x <- as_mlx(x_r, dtype = "uint32")
  expect_equal(mlx_dtype(x), "uint32")
})

test_that("uint64 dtype creation works", {
  x_r <- c(0, 1000, 2^40)
  x <- as_mlx(x_r, dtype = "uint64")
  expect_equal(mlx_dtype(x), "uint64")
})

test_that("integer arithmetic preserves dtype", {
  x <- as_mlx(c(10, 20, 30), dtype = "int32")
  y <- as_mlx(c(1, 2, 3), dtype = "int32")

  # Addition
  z_add <- x + y
  expect_equal(mlx_dtype(z_add), "int32")
  expect_equal(as.vector(z_add), c(11, 22, 33))

  # Subtraction
  z_sub <- x - y
  expect_equal(mlx_dtype(z_sub), "int32")
  expect_equal(as.vector(z_sub), c(9, 18, 27))

  # Multiplication
  z_mul <- x * y
  expect_equal(mlx_dtype(z_mul), "int32")
  expect_equal(as.vector(z_mul), c(10, 40, 90))
})

test_that("integer division behavior", {
  x <- as_mlx(c(10, 20, 30), dtype = "int32")
  y <- as_mlx(c(3, 4, 5), dtype = "int32")

  # Division likely converts to float (MLX behavior)
  z <- x / y
  # Don't assume dtype, just check it works
  expect_s3_class(z, "mlx")

  # Integer division should stay int
  z_floordiv <- x %/% y
  expect_equal(as.vector(z_floordiv), c(3, 5, 6))
})

test_that("mixed integer/float operations promote to float", {
  x_int <- as_mlx(c(10, 20, 30), dtype = "int32")
  y_float <- as_mlx(c(1.5, 2.5, 3.5), dtype = "float32")

  z <- x_int + y_float
  # Should promote to float
  expect_equal(mlx_dtype(z), "float32")
  expect_equal(as.vector(z), c(11.5, 22.5, 33.5))
})

test_that("mixed signed/unsigned integer operations work", {
  x_signed <- as_mlx(c(10, 20, 30), dtype = "int32")
  x_unsigned <- as_mlx(c(1, 2, 3), dtype = "uint32")

  # MLX will promote to appropriate type
  z <- x_signed + x_unsigned
  expect_s3_class(z, "mlx")
  # Don't assume promotion rules, just verify it works
})

test_that("integer reductions work", {
  x <- as_mlx(matrix(1:12, 3, 4), dtype = "int32")

  # Sum
  s <- mlx_sum(x)
  expect_s3_class(s, "mlx")
  expect_equal(as.vector(s), sum(1:12))

  # Mean likely converts to float
  m <- mlx_mean(x)
  expect_s3_class(m, "mlx")
  expect_equal(as.vector(m), mean(1:12))

  # Min/max should work
  min_val <- min(x)
  max_val <- max(x)
  expect_s3_class(min_val, "mlx")
  expect_s3_class(max_val, "mlx")
  expect_equal(as.vector(min_val), 1)
  expect_equal(as.vector(max_val), 12)
})

test_that("integer matrix multiplication limitation", {
  # MLX matmul only supports inexact (floating point) types
  x <- as_mlx(matrix(1:6, 2, 3), dtype = "int32")
  y <- as_mlx(matrix(1:6, 3, 2), dtype = "int32")

  # This should error with informative message
  expect_error(
    x %*% y,
    "Only inexact types are supported"
  )

  # Workaround: convert to float first
  x_float <- as_mlx(matrix(1:6, 2, 3), dtype = "float32")
  y_float <- as_mlx(matrix(1:6, 3, 2), dtype = "float32")
  z <- x_float %*% y_float

  expect_equal(mlx_dim(z), c(2, 2))
  z_r <- matrix(1:6, 2, 3) %*% matrix(1:6, 3, 2)
  expect_equal(as.matrix(z), z_r, tolerance = 1e-6)
})

test_that("integer comparisons work", {
  x <- as_mlx(c(1, 2, 3, 4, 5), dtype = "int32")

  # Greater than
  mask <- x > as_mlx(3, dtype = "int32")
  expect_equal(mlx_dtype(mask), "bool")
  expect_equal(as.vector(mask), c(FALSE, FALSE, FALSE, TRUE, TRUE))

  # Equal
  mask_eq <- x == as_mlx(3, dtype = "int32")
  expect_equal(as.vector(mask_eq), c(FALSE, FALSE, TRUE, FALSE, FALSE))
})

test_that("integer type overflow behavior", {
  # int8 can only hold -128 to 127
  # What happens with overflow depends on MLX behavior
  # Just verify creation works, don't assume overflow semantics

  x <- as_mlx(c(-128, 127), dtype = "int8")
  expect_equal(mlx_dtype(x), "int8")

  # uint8 can only hold 0 to 255
  y <- as_mlx(c(0, 255), dtype = "uint8")
  expect_equal(mlx_dtype(y), "uint8")
})

test_that("conversion from integer mlx to R preserves values", {
  # Small integers should round-trip correctly
  vals <- c(-100, -1, 0, 1, 100)

  for (dtype in c("int8", "int16", "int32", "int64")) {
    x <- as_mlx(vals, dtype = dtype)
    x_r <- as.vector(x)
    expect_equal(x_r, vals, info = paste("dtype =", dtype))
  }

  # Unsigned
  pos_vals <- c(0, 1, 100, 200)
  for (dtype in c("uint8", "uint16", "uint32", "uint64")) {
    x <- as_mlx(pos_vals, dtype = dtype)
    x_r <- as.vector(x)
    expect_equal(x_r, pos_vals, info = paste("dtype =", dtype))
  }
})

test_that("integer types work with mlx_zeros and mlx_ones", {
  # zeros
  z <- mlx_zeros(c(3, 4), dtype = "int32")
  expect_equal(mlx_dtype(z), "int32")
  expect_equal(as.vector(as.matrix(z)), rep(0, 12))

  # ones
  o <- mlx_ones(c(3, 4), dtype = "int32")
  expect_equal(mlx_dtype(o), "int32")
  expect_equal(as.vector(as.matrix(o)), rep(1, 12))

  # uint8
  z_u8 <- mlx_zeros(10, dtype = "uint8")
  expect_equal(mlx_dtype(z_u8), "uint8")
  expect_equal(as.vector(z_u8), rep(0, 10))
})

test_that("integer types work with mlx_arange", {
  # Create integer sequence (mlx_arange(stop, start, ...))
  x <- mlx_arange(10, start = 0, dtype = "int32")
  expect_equal(mlx_dtype(x), "int32")
  expect_equal(as.vector(x), 0:9)

  # With step
  x_step <- mlx_arange(20, start = 0, step = 2, dtype = "int32")
  expect_equal(mlx_dtype(x_step), "int32")
  expect_equal(as.vector(x_step), seq(0, 18, by = 2))
})
