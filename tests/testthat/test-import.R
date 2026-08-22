import_add_matrix_fixture <- function() {
  fixture_names <- c(
    "add_matrix.mlxfn",
    "add_matrix_pre_metadata.mlxfn"
  )
  errors <- character()

  for (fixture_name in fixture_names) {
    fn_path <- system.file("extdata", fixture_name, package = "Rmlx")
    imported <- tryCatch(
      mlx_import_function(fn_path),
      error = function(err) {
        errors <<- c(errors, conditionMessage(err))
        NULL
      }
    )
    if (!is.null(imported)) return(imported)
  }

  stop(
    "No bundled .mlxfn fixture is compatible with this MLX version: ",
    paste(unique(errors), collapse = "; "),
    call. = FALSE
  )
}

test_that("mlx_import_function loads and runs positional args", {
  skip_if_not(mlx_has_gpu())
  imported <- import_add_matrix_fixture()

  a <- mlx_matrix(1:4, 2, 2, dtype = "float32")
  b <- mlx_matrix(5:8, 2, 2, dtype = "float32")

  result <- imported(a, b)
  expect_s3_class(result, "mlx")
  expect_equal(as.matrix(result), as.matrix(a) + as.matrix(b), tolerance = 1e-6)
})

test_that("mlx_import_function accepts named arguments", {
  skip_if_not(mlx_has_gpu())
  imported <- import_add_matrix_fixture()

  a <- mlx_matrix(rep(2, 4), 2, 2, dtype = "float32")
  b <- mlx_matrix(rep(1, 4), 2, 2, dtype = "float32")

  result <- imported(b = b, a = a)
  expect_equal(as.matrix(result), as.matrix(a) + as.matrix(b), tolerance = 1e-6)
})
