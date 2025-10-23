test_that("mlx_relu applies ReLU activation", {
  relu <- mlx_relu()

  x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))
  y <- mlx_forward(relu, x)

  expected <- matrix(c(0, 0, 0, 1, 2), 5, 1)
  expect_equal(as.matrix(y), expected, tolerance = 1e-6)
})

test_that("mlx_relu is an mlx_module with no parameters", {
  relu <- mlx_relu()

  expect_s3_class(relu, "mlx_module")
  expect_s3_class(relu, "mlx_relu")
  expect_equal(length(mlx_parameters(relu)), 0)
})

test_that("mlx_sequential composes modules", {
  set.seed(42)
  layer1 <- mlx_linear(2, 3, bias = FALSE)
  relu <- mlx_relu()
  layer2 <- mlx_linear(3, 1, bias = FALSE)

  net <- mlx_sequential(layer1, relu, layer2)

  expect_s3_class(net, "mlx_sequential")
  expect_s3_class(net, "mlx_module")
  expect_equal(length(net$layers), 3)
})

test_that("mlx_sequential forward pass works", {
  set.seed(123)
  layer1 <- mlx_linear(2, 3, bias = FALSE)
  relu <- mlx_relu()
  layer2 <- mlx_linear(3, 1, bias = FALSE)

  net <- mlx_sequential(layer1, relu, layer2)

  x <- as_mlx(matrix(c(1, 2), 1, 2))
  y <- mlx_forward(net, x)

  expect_s3_class(y, "mlx")
  expect_equal(dim(y), c(1L, 1L))
})

test_that("mlx_sequential collects parameters from all layers", {
  set.seed(456)
  layer1 <- mlx_linear(2, 3, bias = TRUE)
  relu <- mlx_relu()
  layer2 <- mlx_linear(3, 1, bias = TRUE)

  net <- mlx_sequential(layer1, relu, layer2)
  params <- mlx_parameters(net)

  # layer1 has weight and bias (2), relu has none (0), layer2 has weight and bias (2)
  expect_equal(length(params), 4)
})

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
