test_that("mlx_metal_kernel runs a simple elementwise kernel", {
  skip_if_not(mlx_has_gpu())

  add_one <- mlx_metal_kernel(
    name = "add_one_test",
    input_names = "inp",
    output_names = "out",
    source = "
      uint elem = thread_position_in_grid.x;
      out[elem] = inp[elem] + (T)1;
    "
  )

  x <- mlx_cast(as_mlx(1:8), "float32")
  out <- add_one(
    inputs = list(x),
    output_shapes = list(c(length(x))),
    output_dtypes = "float32",
    grid = c(length(x), 1L, 1L),
    threadgroup = c(length(x), 1L, 1L),
    template = list(T = "float32")
  )

  expect_equal(as.numeric(out), 2:9, tolerance = 1e-6)
})

test_that("mlx_metal_kernel validates launch parameters", {
  skip_if_not(mlx_has_gpu())

  add_one <- mlx_metal_kernel(
    name = "add_one_validate",
    input_names = "inp",
    output_names = "out",
    source = "
      uint elem = thread_position_in_grid.x;
      out[elem] = inp[elem] + (T)1;
    "
  )

  x <- mlx_cast(as_mlx(1:4), "float32")

  expect_error(
    add_one(
      inputs = list(x),
      output_shapes = list(c(length(x))),
      output_dtypes = "float32",
      grid = c(length(x), 1L),
      threadgroup = c(length(x), 1L, 1L),
      template = list(T = "float32")
    ),
    "length 3"
  )

  expect_error(
    add_one(
      inputs = list(x),
      output_shapes = list(c(length(x))),
      output_dtypes = "float32",
      grid = c(length(x), 1L, 1L),
      threadgroup = c(length(x), 1L, 1L),
      template = list(T = 1.5)
    ),
    "whole number"
  )
})
