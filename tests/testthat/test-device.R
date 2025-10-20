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
