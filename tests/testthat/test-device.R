test_that("mlx_device getter works", {
  old_device <- mlx_device()
  expect_type(old_device, "character")
  expect_true(old_device %in% c("gpu", "cpu"))
})

test_that("mlx_device setter works", {
  skip_if_not(mlx_has_gpu())
  old_device <- mlx_device()
  on.exit(mlx_device(old_device), add = TRUE)

  mlx_device("cpu")

  mlx_device("gpu")
})

test_that("with_device temporarily overrides device", {
  original <- mlx_device()
  on.exit(mlx_device(original), add = TRUE)

  result <- with_device("cpu", {
    "value"
  })

  expect_equal(result, "value")
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
  original <- mlx_device()
  on.exit(mlx_device(original), add = TRUE)

  fn <- function() {
    local_device("cpu")
  }
  fn()
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
