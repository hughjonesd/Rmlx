ends_with_case <- function(path, ext) {
  if (nchar(path) < nchar(ext)) {
    return(FALSE)
  }
  tolower(substr(path, nchar(path) - nchar(ext) + 1L, nchar(path))) == tolower(ext)
}

test_that("mlx_save and mlx_load round trip arrays", {
  base_path <- tempfile(fileext = ".mlx")

  x <- matrix(rnorm(9), 3, 3)
  tensor <- as_mlx(x, device = "cpu")

  saved_path <- mlx_save(tensor, base_path)
  on.exit(unlink(saved_path, recursive = TRUE), add = TRUE)

  expect_true(file.exists(saved_path))
  expect_equal(saved_path, paste0(path.expand(base_path), ".npy"))

  loaded <- mlx_load(base_path, device = "cpu")
  expect_s3_class(loaded, "mlx")
  expect_identical(loaded$device, "cpu")
  expect_equal(as.matrix(loaded), x, tolerance = 1e-6)
})

test_that("mlx_load validates file path", {
  missing_path <- tempfile(fileext = ".mlx")
  expected <- paste0(path.expand(missing_path), ".npy")
  expect_false(file.exists(expected))
  expect_error(mlx_load(missing_path), expected, fixed = TRUE)
})

test_that("mlx_save_safetensors round trips tensors and metadata", {
  base_path <- tempfile(fileext = ".safetensors")

  tensors <- list(
    mat = as_mlx(matrix(rnorm(4), 2, 2), device = "cpu"),
    vec = as_mlx(1:3, device = "cpu")
  )
  metadata <- c(author = "rmlx", version = "1.0")

  saved_path <- mlx_save_safetensors(base_path, tensors, metadata)
  on.exit(unlink(saved_path, recursive = TRUE), add = TRUE)

  expect_true(file.exists(saved_path))
  expected_path <- if (ends_with_case(base_path, ".safetensors")) {
    path.expand(base_path)
  } else {
    paste0(path.expand(base_path), ".safetensors")
  }
  expect_equal(saved_path, expected_path)

  loaded <- mlx_load_safetensors(base_path, device = "cpu")
  expect_named(loaded$tensors, names(tensors))
  expect_equal(as.matrix(loaded$tensors$mat), as.matrix(tensors$mat), tolerance = 1e-6)
  expect_equal(as.vector(as.matrix(loaded$tensors$vec)), as.vector(as.matrix(tensors$vec)))
  expect_named(loaded$metadata, names(metadata))
  expect_equal(unname(loaded$metadata), unname(metadata))
})

test_that("mlx_save_gguf handles mixed metadata", {
  base_path <- tempfile(fileext = ".gguf")

  tensors <- list(emb = as_mlx(matrix(1:4, 2, 2), device = "cpu"))
  metadata <- list(
    description = "demo",
    tags = c("a", "b"),
    scale = as_mlx(1:2, device = "cpu")
  )

  saved_path <- mlx_save_gguf(base_path, tensors, metadata)
  on.exit(unlink(saved_path, recursive = TRUE), add = TRUE)

  expect_true(file.exists(saved_path))
  expected_path <- if (ends_with_case(base_path, ".gguf")) {
    path.expand(base_path)
  } else {
    paste0(path.expand(base_path), ".gguf")
  }
  expect_equal(saved_path, expected_path)

  loaded <- mlx_load_gguf(base_path, device = "cpu")
  expect_named(loaded$tensors, names(tensors))
  expect_equal(as.matrix(loaded$tensors$emb), as.matrix(tensors$emb))

  expect_setequal(names(loaded$metadata), names(metadata))
  expect_equal(loaded$metadata$description, metadata$description)
  expect_equal(loaded$metadata$tags, metadata$tags)
  expect_s3_class(loaded$metadata$scale, "mlx")
  expect_equal(as.vector(as.matrix(loaded$metadata$scale)), as.vector(as.matrix(metadata$scale)))
})
