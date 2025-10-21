test_that("sgd training reduces linear regression loss", {
  set.seed(123)
  n <- 40
  in_features <- 3
  out_features <- 1

  x_mat <- matrix(rnorm(n * in_features), n, in_features)
  true_w <- matrix(c(2, -1, 0.5), in_features, out_features)
  y_mat <- x_mat %*% true_w

  x <- as_mlx(x_mat)
  y <- as_mlx(y_mat)

  model <- mlx_linear(in_features, out_features, bias = FALSE)
  params <- mlx_parameters(model)
  opt <- mlx_optimizer_sgd(params, lr = 0.05)

  loss_fn <- function(module, data_x, data_y) {
    preds <- mlx_forward(module, data_x)
    resids <- preds - data_y
    sum(resids * resids) / length(data_y)
  }

  loss_value <- function(module, data_x, data_y) {
    as.numeric(sum((as.matrix(mlx_forward(module, data_x)) - as.matrix(data_y))^2) / n)
  }

  initial_loss <- loss_value(model, x, y)

  for (step in 1:50) {
    mlx_train_step(model, loss_fn, opt, x, y)
  }

  final_loss <- loss_value(model, x, y)

  expect_lt(final_loss, initial_loss)
  expect_true(final_loss < 0.1)
})
