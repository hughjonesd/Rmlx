test_that("activation functions work correctly", {
  # Tests assume MLX is available

  x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))

  # GELU
  gelu <- mlx_gelu()
  result <- mlx_forward(gelu, x)
  expect_s3_class(result, "mlx")
  expect_equal(dim(result), c(5L, 1L))
  # GELU(0) should be 0
  expect_equal(as.numeric(as.matrix(result)[3, 1]), 0, tolerance = 1e-6)

  # Sigmoid
  sigmoid <- mlx_sigmoid()
  result <- mlx_forward(sigmoid, x)
  expect_s3_class(result, "mlx")
  # Sigmoid outputs should be in (0, 1)
  result_vals <- as.numeric(as.matrix(result))
  expect_true(all(result_vals > 0 & result_vals < 1))
  # Sigmoid(0) = 0.5
  expect_equal(result_vals[3], 0.5, tolerance = 1e-6)

  # Tanh
  tanh_layer <- mlx_tanh()
  result <- mlx_forward(tanh_layer, x)
  expect_s3_class(result, "mlx")
  # Tanh outputs should be in (-1, 1)
  result_vals <- as.numeric(as.matrix(result))
  expect_true(all(result_vals > -1 & result_vals < 1))
  # Tanh(0) = 0
  expect_equal(result_vals[3], 0, tolerance = 1e-6)

  # Leaky ReLU
  lrelu <- mlx_leaky_relu(negative_slope = 0.1)
  result <- mlx_forward(lrelu, x)
  expect_s3_class(result, "mlx")
  result_vals <- as.numeric(as.matrix(result))
  # Positive values unchanged
  expect_equal(result_vals[5], 2, tolerance = 1e-6)
  # Negative values scaled by slope
  expect_equal(result_vals[1], -2 * 0.1, tolerance = 1e-6)

  # SiLU
  silu <- mlx_silu()
  result <- mlx_forward(silu, x)
  expect_s3_class(result, "mlx")
  # SiLU(0) = 0
  expect_equal(as.numeric(as.matrix(result)[3, 1]), 0, tolerance = 1e-6)

  # Softmax
  softmax <- mlx_softmax_layer()
  x_multi <- as_mlx(matrix(1:6, 2, 3))
  result <- mlx_forward(softmax, x_multi)
  expect_s3_class(result, "mlx")
  # Rows should sum to 1
  row_sums <- rowSums(as.matrix(result))
  expect_equal(row_sums, c(1, 1), tolerance = 1e-6)
})

test_that("dropout layer works correctly", {
  # Tests assume MLX is available

  set.seed(42)
  dropout <- mlx_dropout(p = 0.5)
  x <- as_mlx(matrix(1:12, 3, 4))

  # In training mode, some values should be zeroed
  result <- mlx_forward(dropout, x)
  expect_s3_class(result, "mlx")
  expect_equal(dim(result), dim(x))

  # Set to eval mode
  mlx_set_training(dropout, FALSE)
  result_eval <- mlx_forward(dropout, x)
  # In eval mode, output should equal input
  expect_equal(as.matrix(result_eval), as.matrix(x), tolerance = 1e-6)

  # Test p = 0 (no dropout)
  dropout0 <- mlx_dropout(p = 0)
  result <- mlx_forward(dropout0, x)
  expect_equal(as.matrix(result), as.matrix(x), tolerance = 1e-6)

  # Test p = 1 (all dropout)
  dropout1 <- mlx_dropout(p = 1)
  result <- mlx_forward(dropout1, x)
  expect_equal(as.matrix(result), matrix(0, 3, 4), tolerance = 1e-6)
})

test_that("layer normalization works correctly", {
  # Tests assume MLX is available

  set.seed(1)
  ln <- mlx_layer_norm(normalized_shape = 4)
  x <- as_mlx(matrix(rnorm(12), 3, 4))

  result <- mlx_forward(ln, x)
  expect_s3_class(result, "mlx")
  expect_equal(dim(result), c(3L, 4L))

  # Each row should have approximately mean 0 and std 1
  result_mat <- as.matrix(result)
  row_means <- rowMeans(result_mat)
  # Use ddof=0 to match the normalization (population std, not sample std)
  row_sds <- apply(result_mat, 1, function(x) sqrt(mean((x - mean(x))^2)))
  expect_equal(row_means, c(0, 0, 0), tolerance = 1e-5)
  expect_equal(row_sds, c(1, 1, 1), tolerance = 1e-4)

  # Check parameters exist
  params <- mlx_parameters(ln)
  expect_length(params, 2)
})

test_that("batch normalization works correctly", {
  # Tests assume MLX is available

  set.seed(1)
  bn <- mlx_batch_norm(num_features = 4)
  x <- as_mlx(matrix(rnorm(12), 3, 4))

  result <- mlx_forward(bn, x)
  expect_s3_class(result, "mlx")
  expect_equal(dim(result), c(3L, 4L))

  # In training mode, each column should be normalized
  result_mat <- as.matrix(result)
  col_means <- colMeans(result_mat)
  expect_equal(col_means, c(0, 0, 0, 0), tolerance = 1e-5)

  # Check parameters exist
  params <- mlx_parameters(bn)
  expect_length(params, 2)

  # Test eval mode
  mlx_set_training(bn, FALSE)
  result_eval <- mlx_forward(bn, x)
  expect_s3_class(result_eval, "mlx")
})

test_that("embedding layer works correctly", {
  # Tests assume MLX is available

  set.seed(1)
  vocab_size <- 100
  embed_dim <- 16
  emb <- mlx_embedding(num_embeddings = vocab_size, embedding_dim = embed_dim)

  # Test single token
  token <- as_mlx(5)
  result <- mlx_forward(emb, token)
  expect_s3_class(result, "mlx")
  expect_equal(dim(result), c(1L, embed_dim))

  # Test multiple tokens
  tokens <- as_mlx(c(1, 5, 10, 3))
  result <- mlx_forward(emb, tokens)
  expect_s3_class(result, "mlx")
  expect_equal(dim(result), c(4L, embed_dim))

  # Test 2D input
  tokens_2d <- as_mlx(matrix(c(1, 5, 10, 3), 2, 2))
  result <- mlx_forward(emb, tokens_2d)
  expect_s3_class(result, "mlx")
  expect_equal(dim(result), c(2L, 2L, embed_dim))

  # Check parameters
  params <- mlx_parameters(emb)
  expect_length(params, 1)

  # Test out of bounds
  expect_error(mlx_forward(emb, as_mlx(100)), "out of range")
  expect_error(mlx_forward(emb, as_mlx(-1)), "out of range")
})

test_that("activation modules have no parameters", {
  # Tests assume MLX is available

  activations <- list(
    mlx_gelu(),
    mlx_sigmoid(),
    mlx_tanh(),
    mlx_leaky_relu(),
    mlx_silu(),
    mlx_softmax_layer(),
    mlx_dropout()
  )

  for (act in activations) {
    params <- mlx_parameters(act)
    expect_length(params, 0)
  }
})

test_that("modules work in sequential", {
  # Tests assume MLX is available

  set.seed(1)
  net <- mlx_sequential(
    mlx_linear(4, 8),
    mlx_gelu(),
    mlx_dropout(p = 0.2),
    mlx_linear(8, 2),
    mlx_softmax_layer()
  )

  x <- as_mlx(matrix(rnorm(12), 3, 4))
  result <- mlx_forward(net, x)

  expect_s3_class(result, "mlx")
  expect_equal(dim(result), c(3L, 2L))

  # Output should be probabilities (sum to 1)
  result_mat <- as.matrix(result)
  row_sums <- rowSums(result_mat)
  expect_equal(row_sums, c(1, 1, 1), tolerance = 1e-5)

  # Check parameters from all layers
  params <- mlx_parameters(net)
  expect_length(params, 4) # 2 weights + 2 biases from linear layers
})

test_that("mlx_set_training propagates through sequential modules", {
  net <- mlx_sequential(
    mlx_dropout(p = 0.4),
    mlx_batch_norm(num_features = 2)
  )

  mlx_set_training(net, FALSE)

  expect_false(net$layers[[1]]$.env$training)
  expect_false(net$layers[[2]]$.env$training)
})
