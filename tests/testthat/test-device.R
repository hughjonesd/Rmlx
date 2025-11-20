test_that("mlx_default_device getter works", {
  old_device <- mlx_default_device()
  expect_type(old_device, "character")
  expect_true(old_device %in% c("gpu", "cpu"))
})

test_that("mlx_default_device setter works", {
  old_device <- mlx_default_device()

  mlx_default_device("cpu")
  expect_equal(mlx_default_device(), "cpu")

  mlx_default_device("gpu")
  expect_equal(mlx_default_device(), "gpu")

  # Restore original
  mlx_default_device(old_device)
})

test_that("device argument is respected", {
  x <- matrix(1:12, 3, 4)

  x_gpu <- as_mlx(x, device = "gpu")
  expect_equal(x_gpu$device, "gpu")

  x_cpu <- as_mlx(x, device = "cpu")
  expect_equal(x_cpu$device, "cpu")
})

test_that("with_default_device temporarily overrides device", {
  original <- mlx_default_device()
  on.exit(mlx_default_device(original), add = TRUE)

  result <- with_default_device("cpu", {
    expect_equal(mlx_default_device(), "cpu")
    "value"
  })

  expect_equal(result, "value")
  expect_equal(mlx_default_device(), original)
})

test_that("mlx_best_device returns a valid device", {
  device <- mlx_best_device()
  expect_type(device, "character")
  expect_true(device %in% c("gpu", "cpu"))
})

test_that("mlx_best_device returns gpu when available", {
  # This test assumes the system has a GPU
  # On systems without GPU, mlx_best_device() should return "cpu"
  device <- mlx_best_device()
  has_gpu <- mlx_has_gpu()

  if (has_gpu) {
    expect_equal(device, "gpu")
  } else {
    expect_equal(device, "cpu")
  }
})

test_that("mlx_best_device is consistent with mlx_has_gpu", {
  # mlx_best_device should return "gpu" iff mlx_has_gpu is TRUE
  device <- mlx_best_device()
  has_gpu <- mlx_has_gpu()

  expect_equal(device == "gpu", has_gpu)
})

test_that("mlx_device returns device of mlx object", {
  # Create object on GPU
  x_gpu <- as_mlx(1:10, device = "gpu")
  expect_equal(mlx_device(x_gpu), "gpu")

  # Create object on CPU
  x_cpu <- as_mlx(1:10, device = "cpu")
  expect_equal(mlx_device(x_cpu), "cpu")
})

test_that("mlx_device works with different object types", {
  # Vector
  vec <- as_mlx(1:5, device = "cpu")
  expect_equal(mlx_device(vec), "cpu")

  # Matrix
  mat <- mlx_matrix(1:12, 3, 4, device = "gpu")
  expect_equal(mlx_device(mat), "gpu")

  # Array
  arr <- mlx_array(1:24, c(2, 3, 4), device = "cpu")
  expect_equal(mlx_device(arr), "cpu")
})

test_that("mlx_device errors on non-mlx input", {
  expect_error(mlx_device(1:10), "is_mlx\\(x\\) is not TRUE")
  expect_error(mlx_device(matrix(1:9, 3, 3)), "is_mlx\\(x\\) is not TRUE")
})
