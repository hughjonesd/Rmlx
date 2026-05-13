test_that("mlx_device getter works", {
  old_device <- mlx_device()
  expect_type(old_device, "character")
  expect_true(old_device %in% c("gpu", "cpu"))
})

test_that("mlx_device setter works", {
  skip_if_not(mlx_has_gpu())
  local_device(mlx_device())

  mlx_device("cpu")
  expect_equal(mlx_device(), "cpu")

  mlx_device("gpu")
  expect_equal(mlx_device(), "gpu")
})

test_that("with_device temporarily overrides device", {
  skip_if_not(mlx_has_gpu())
  local_device(mlx_device())

  mlx_device("gpu")

  result <- with_device("cpu", {
    expect_equal(mlx_device(), "cpu")
    "value"
  })

  expect_equal(result, "value")
  expect_equal(mlx_device(), "gpu")
})

test_that("operations respect the current device", {
  skip_if_not(mlx_has_gpu())

  x <- with_device("cpu", as_mlx(c(1, 2, 3), dtype = "float64"))
  m <- with_device("cpu", as_mlx(matrix(c(4, 1, 1, 3), 2, 2), dtype = "float64"))

  with_device("cpu", {
    expect_equal(as.vector(x + 1), c(2, 3, 4), tolerance = 1e-12)
    expect_equal(as.vector(sin(x)), sin(c(1, 2, 3)), tolerance = 1e-6)
    expect_equal(as.numeric(mlx_sum(x)), 6, tolerance = 1e-12)
    expect_equal(as.matrix(m %*% m), matrix(c(17, 7, 7, 10), 2, 2), tolerance = 1e-12)
  })

  with_device("gpu", {
    expect_error(as.vector(x + 1), "float64")
    expect_error(as.vector(sin(x)), "float64")
    expect_error(as.numeric(mlx_sum(x)), "float64")
    expect_error(as.matrix(m %*% m), "float64")
  })
})

test_that("with_device accepts streams", {
  stream <- mlx_new_stream("cpu")
  original_device <- mlx_device()
  original_stream <- mlx_default_stream("cpu")
  on.exit({
    mlx_set_default_stream(original_stream)
    mlx_device(original_device)
  }, add = TRUE)

  result <- with_device(stream, {
    current <- mlx_default_stream(stream$device)
    expect_equal(current$index, stream$index)
    "ok"
  })

  expect_equal(result, "ok")
  restored <- mlx_default_stream(stream$device)
  expect_equal(restored$index, original_stream$index)
})

test_that("local_device restores device", {
  skip_if_not(mlx_has_gpu())
  local_device(mlx_device())

  mlx_device("gpu")

  fn <- function() {
    local_device("cpu")
    expect_equal(mlx_device(), "cpu")
  }
  fn()
  expect_equal(mlx_device(), "gpu")
})

test_that("local_device accepts streams", {
  stream <- mlx_new_stream("cpu")
  original_device <- mlx_device()
  original_stream <- mlx_default_stream(stream$device)
  on.exit({
    mlx_set_default_stream(original_stream)
    mlx_device(original_device)
  }, add = TRUE)

  fn <- function() {
    local_device(stream)
    current <- mlx_default_stream(stream$device)
    expect_equal(current$index, stream$index)
  }

  fn()
  restored <- mlx_default_stream(stream$device)
  expect_equal(restored$index, original_stream$index)
})

test_that("mlx_best_device returns a valid device", {
  device <- mlx_best_device()
  expect_type(device, "character")
  expect_true(device %in% c("gpu", "cpu"))
})

test_that("mlx_best_device returns gpu when available", {
  device <- mlx_best_device()
  has_gpu <- mlx_has_gpu()

  if (has_gpu) {
    expect_equal(device, "gpu")
  } else {
    expect_equal(device, "cpu")
  }
})

test_that("mlx_best_device is consistent with mlx_has_gpu", {
  device <- mlx_best_device()
  has_gpu <- mlx_has_gpu()

  expect_equal(device == "gpu", has_gpu)
})
