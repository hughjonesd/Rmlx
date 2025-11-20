test_that("mlx_import_function loads and runs positional args", {
  fn_path <- system.file("extdata", "add_matrix.mlxfn", package = "Rmlx")
  imported <- mlx_import_function(fn_path, device = "cpu")

  a <- mlx_matrix(1:4, 2, 2, dtype = "float32", device = "cpu")
  b <- mlx_matrix(5:8, 2, 2, dtype = "float32", device = "cpu")

  result <- imported(a, b)
  expect_s3_class(result, "mlx")
  expect_equal(as.matrix(result), as.matrix(a) + as.matrix(b), tolerance = 1e-6)
})

test_that("mlx_import_function accepts named arguments", {
  fn_path <- system.file("extdata", "add_matrix.mlxfn", package = "Rmlx")
  imported <- mlx_import_function(fn_path, device = "cpu")

  a <- mlx_matrix(rep(2, 4), 2, 2, dtype = "float32", device = "cpu")
  b <- ml_matrix(rep(1, 4), 2, 2, dtype = "float32", device = "cpu")

  result <- imported(b = b, a = a)
  expect_equal(as.matrix(result), as.matrix(a) + as.matrix(b), tolerance = 1e-6)
})
