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

test_that("mixed-device arithmetic stages to GPU and preserves inputs", {
  skip_if_not(mlx_has_gpu())

  cpu_mat <- mlx_matrix(1:6, 2, 3, device = "cpu", dtype = "float32")
  gpu_mat <- mlx_matrix(rep(2, 6), 2, 3, device = "gpu", dtype = "float32")

  res <- cpu_mat + gpu_mat

  expect_equal(res$device, "gpu")
  expect_equal(as.array(res), as.array(cpu_mat) + as.array(gpu_mat))
  expect_equal(cpu_mat$device, "cpu")
  expect_equal(gpu_mat$device, "gpu")
})

test_that("mixed-device reductions with multiple operands choose GPU", {
  skip_if_not(mlx_has_gpu())

  cpu_mat <- mlx_matrix(1:6, 2, 3, device = "cpu", dtype = "float32")
  gpu_mat <- mlx_matrix(7:12, 2, 3, device = "gpu", dtype = "float32")

  res <- sum(cpu_mat, gpu_mat)

  expect_equal(res$device, "gpu")
  expect_equal(as.array(res), sum(as.array(cpu_mat)) + sum(as.array(gpu_mat)))
})

test_that("requesting GPU errors when backend is unavailable", {
  skip_if(mlx_has_gpu())

  expect_error(mlx_zeros(c(2, 2), device = "gpu"), "gpu")
  expect_error(as_mlx(1:3, device = "gpu"), "gpu")
})
