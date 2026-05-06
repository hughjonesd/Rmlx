local_gpu_default <- function() {
  skip_if_not(mlx_has_gpu())
  old_device <- mlx_default_device()
  mlx_default_device("gpu")
  old_device
}

force_values <- function(x) {
  mlx_eval(x)
  as_r(x)
}

test_that("ordering helpers use the array preferred device, not the global default", {
  old_device <- local_gpu_default()
  on.exit(mlx_default_device(old_device), add = TRUE)

  # Bool and complex64 ordering work on CPU but not through the current Metal
  # kernels. With global default GPU, failures here show that the operation was
  # scheduled on the global default rather than on x$device.
  bool_x <- as_mlx(c(TRUE, FALSE, TRUE, FALSE), dtype = "bool", device = "cpu")
  complex_x <- as_mlx(c(1 + 1i, 0 + 2i, 3 + 0i, 2 + 0i),
                      dtype = "complex64", device = "cpu")

  expect_equal(mlx_device(bool_x), "cpu")
  expect_equal(mlx_device(complex_x), "cpu")

  expect_no_error(force_values(mlx_sort(bool_x)))
  expect_no_error(force_values(mlx_argsort(complex_x)))
})
