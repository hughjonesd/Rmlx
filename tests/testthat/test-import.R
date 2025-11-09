test_that("mlx_import_function loads and runs positional args", {
  fn_path <- testthat::test_path("fixtures", "add_matrix.mlxfn")
  imported <- mlx_import_function(fn_path, device = "cpu")

  a <- as_mlx(matrix(1:4, 2, 2), dtype = "float32", device = "cpu")
  b <- as_mlx(matrix(5:8, 2, 2), dtype = "float32", device = "cpu")

  result <- imported(a, b)
  expect_s3_class(result, "mlx")
  expect_equal(as.matrix(result), as.matrix(a) + as.matrix(b), tolerance = 1e-6)
})

test_that("mlx_import_function accepts named arguments", {
  fn_path <- testthat::test_path("fixtures", "add_matrix.mlxfn")
  imported <- mlx_import_function(fn_path, device = "cpu")

  a <- as_mlx(matrix(2, 2, 2), dtype = "float32", device = "cpu")
  b <- as_mlx(matrix(1, 2, 2), dtype = "float32", device = "cpu")

  result <- imported(b = b, a = a)
  expect_equal(as.matrix(result), as.matrix(a) + as.matrix(b), tolerance = 1e-6)
})
