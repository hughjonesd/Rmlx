test_that("mlx_grad accepts length-1 vector outputs", {
  w <- mlx_vector(c(2, 4, 6), device = "cpu")

  loss_fn <- function(x) {
    # Returns a 1D length-1 mlx vector, not a scalar
    x[1]
  }

  grad <- mlx_grad(loss_fn, w)[[1]]

  expect_equal(as.vector(grad), c(1, 0, 0))
  expect_equal(mlx_shape(grad), mlx_shape(w))
})
