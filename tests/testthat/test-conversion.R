test_that("conversion retains suitable R payload storage", {
  for (dtype in c("float32", "float64", "int8", "int16", "int32", "int64",
                  "uint8", "uint16", "uint32", "uint64")) {
    expect_identical(coerce_payload(c(0L, 1L, 2L), dtype), c(0L, 1L, 2L))
    expect_identical(coerce_payload(c(TRUE, FALSE), dtype), c(TRUE, FALSE))
    expect_identical(coerce_payload(c(0, 1, 2), dtype), c(0, 1, 2))
  }
  mask <- matrix(c(TRUE, FALSE), 2, 3)
  expect_identical(coerce_payload(mask, "bool"), mask)
  expect_identical(coerce_payload(c(TRUE, FALSE), "bool"), c(TRUE, FALSE))
  expect_identical(coerce_payload(c(0, 2, -1), "bool"), c(FALSE, TRUE, TRUE))

  # Missing integer/logical values must reach MLX as doubles, not the R sentinel.
  for (input in list(c(1L, NA_integer_), c(TRUE, NA))) {
    expect_identical(coerce_payload(input, "float32"), as.numeric(input))
    expect_identical(coerce_payload(input, "int64"), as.numeric(input))
    expect_error(coerce_payload(input, "bool"), "Logical NA values")
  }
  expect_identical(coerce_payload(factor(c("a", "b")), "int32"), c(1, 2))
  expect_identical(coerce_payload(1:2, "complex64"), c(1 + 0i, 2 + 0i))
})

test_that("direct conversion preserves dtype, layout and dimnames", {
  local_device(mlx_device())
  for (device in c("cpu", "gpu")) {
    mlx_device(device)
    for (dtype in c("int8", "int16", "int32", "int64",
                    "uint8", "uint16", "uint32", "uint64", "bool")) {
      for (input in list(0:11, as.numeric(0:11), rep(c(TRUE, FALSE), 6))) {
        expected <- if (dtype == "bool") as.logical(input) else as.numeric(input)
        expect_equal(as.vector(as_mlx(input, dtype)), expected)
        expect_equal(mlx_dtype(as_mlx(input, dtype)), dtype)

        dim(input) <- c(2L, 3L, 2L)
        dimnames(input) <- list(c("a", "b"), c("x", "y", "z"), c("u", "v"))
        dim(expected) <- dim(input)
        dimnames(expected) <- dimnames(input)
        expect_equal(as.array(as_mlx(input, dtype)), expected)
        expect_equal(
          as.array(mlx_array(input, dim(input), dtype, dimnames(input))),
          expected
        )
      }
    }
  }
})

test_that("direct conversion handles integer limits and fractional doubles", {
  values <- list(
    int8 = c(-128, -1, 0, 127),
    int16 = c(-32768, -1, 0, 32767),
    int32 = c(-2147483647, -1, 0, 2147483647),
    int64 = c(-2^40, -1, 0, 2^40),
    uint8 = c(0, 1, 255),
    uint16 = c(0, 1, 65535),
    uint32 = c(0, 1, 2^32 - 1),
    uint64 = c(0, 1, 2^40)
  )
  local_device(mlx_device())
  for (device in c("cpu", "gpu")) {
    mlx_device(device)
    for (dtype in names(values)) {
      input <- values[[dtype]]
      expect_equal(as.vector(as_mlx(input, dtype)), input)
      if (all(abs(input) <= .Machine$integer.max)) {
        expect_equal(as.vector(as_mlx(as.integer(input), dtype)), input)
      }
      expect_equal(as.vector(as_mlx(c(0.9, 1.5, 2.9), dtype)), c(0, 1, 2))
    }
  }
})

test_that("direct conversion handles scalars, empty inputs and recycling", {
  local_device(mlx_device())
  for (device in c("cpu", "gpu")) {
    mlx_device(device)
    for (dtype in c("int8", "int16", "int32", "int64",
                    "uint8", "uint16", "uint32", "uint64", "bool")) {
      for (input in list(1L, 1, TRUE)) {
        scalar <- mlx_array(input, integer(0), dtype)
        expect_identical(mlx_shape(scalar), integer(0))
        expect_equal(as.vector(scalar), if (dtype == "bool") TRUE else 1)

        empty <- as_mlx(input[FALSE], dtype)
        expect_equal(length(empty), 0L)
        expect_equal(mlx_dtype(empty), dtype)
      }
      recycled <- mlx_matrix(c(0L, 1L), 2, 3, dtype = dtype)
      expected <- matrix(c(0, 1), 2, 3)
      if (dtype == "bool") storage.mode(expected) <- "logical"
      expect_equal(as.matrix(recycled), expected)
    }
  }
})

test_that("direct conversion preserves missing values and independent ownership", {
  local_device(mlx_device())
  for (device in c("cpu", "gpu")) {
    mlx_device(device)
    for (input in list(c(1L, NA_integer_), c(TRUE, NA))) {
      for (dtype in c("float32", "float64")) {
        output <- as.vector(as_mlx(input, dtype))
        expect_equal(output[1], 1)
        # float64 retains the R NA bit pattern; float32 conversion yields NaN.
        expect_identical(output[2], if (dtype == "float64") NA_real_ else NaN)
      }
      expect_error(as_mlx(input, "bool"), "Logical NA values")
      expect_error(mlx_array(input, 2, "bool"), "Logical NA values")
    }
    expect_error(as_mlx(c(NaN, Inf), "bool"), "Logical NA values")

    for (dtype in c("int8", "int16", "int32", "int64",
                    "uint8", "uint16", "uint32", "uint64", "bool")) {
      input <- c(0L, 1L, 2L)
      expected <- if (dtype == "bool") c(FALSE, TRUE, TRUE) else c(0, 1, 2)
      output <- as_mlx(input, dtype)
      input[] <- 0L
      expect_equal(as.vector(output), expected)
    }
  }
})
