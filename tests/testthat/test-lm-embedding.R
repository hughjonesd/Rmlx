test_that("embedding preserves multidimensional index layout", {
  layer <- mlx_embedding(4, 3)
  weights <- matrix(seq_len(12), 4, 3)
  mlx_param_set_values(mlx_parameters(layer), list(as_mlx(weights)))
  indices <- matrix(c(1L, 3L, 2L, 4L), 2, 2)
  actual <- as.array(mlx_forward(layer, as_mlx(indices)))
  expected <- array(0, c(2, 2, 3))
  for (i in 1:2) for (j in 1:2) expected[i, j, ] <- weights[indices[i, j], ]
  expect_equal(actual, expected)
})

test_that("embedding gradients accumulate repeated rows", {
  layer <- mlx_embedding(4, 3)
  params <- mlx_parameters(layer)
  loss <- function(w) {
    mlx_param_set_values(params, list(w))
    sum(mlx_forward(layer, as_mlx(c(1L, 1L, 3L))))
  }
  gradient <- mlx_grad(loss, mlx_param_values(params)[[1]])[[1]]
  expect_equal(as.matrix(gradient), matrix(c(2, 0, 1, 0), 4, 3))
})

test_that("embedding validates indices and preserves scalar behavior", {
  layer <- mlx_embedding(4, 3)
  for (indices in list(NA_real_, Inf, 0, 5, 1.5)) {
    expect_error(mlx_forward(layer, as_mlx(indices)), "Index out of range")
  }
  expect_equal(mlx_shape(mlx_forward(layer, mlx_scalar(2L))), c(1L, 3L))
})
