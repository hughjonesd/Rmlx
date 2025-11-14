test_that("MSE loss works correctly", {
  # Tests assume MLX is available

  preds <- as_mlx(matrix(c(1.5, 2.5, 3.5), 3, 1))
  targets <- as_mlx(matrix(c(1, 2, 3), 3, 1))

  # Test mean reduction
  loss <- mlx_mse_loss(preds, targets, reduction = "mean")
  expect_s3_class(loss, "mlx")
  expect_equal(length(dim(loss)), 0L) # scalar
  expected <- mean((c(1.5, 2.5, 3.5) - c(1, 2, 3))^2)
  expect_equal(as.numeric(as.matrix(loss)), expected, tolerance = 1e-6)

  # Test sum reduction
  loss_sum <- mlx_mse_loss(preds, targets, reduction = "sum")
  expect_equal(as.numeric(as.matrix(loss_sum)), expected * 3, tolerance = 1e-6)

  # Test none reduction
  loss_none <- mlx_mse_loss(preds, targets, reduction = "none")
  expect_equal(dim(loss_none), c(3L, 1L))
  expected_none <- (c(1.5, 2.5, 3.5) - c(1, 2, 3))^2
  expect_equal(as.numeric(as.matrix(loss_none)), expected_none, tolerance = 1e-6)

  # Test perfect prediction
  loss_perfect <- mlx_mse_loss(preds, preds)
  expect_equal(as.numeric(as.matrix(loss_perfect)), 0, tolerance = 1e-6)
})

test_that("L1 loss works correctly", {
  # Tests assume MLX is available

  preds <- as_mlx(matrix(c(1.5, 2.5, 3.5), 3, 1))
  targets <- as_mlx(matrix(c(1, 2, 3), 3, 1))

  # Test mean reduction
  loss <- mlx_l1_loss(preds, targets, reduction = "mean")
  expect_s3_class(loss, "mlx")
  expected <- mean(abs(c(1.5, 2.5, 3.5) - c(1, 2, 3)))
  expect_equal(as.numeric(as.matrix(loss)), expected, tolerance = 1e-6)

  # Test sum reduction
  loss_sum <- mlx_l1_loss(preds, targets, reduction = "sum")
  expect_equal(as.numeric(as.matrix(loss_sum)), expected * 3, tolerance = 1e-6)

  # Test none reduction
  loss_none <- mlx_l1_loss(preds, targets, reduction = "none")
  expect_equal(dim(loss_none), c(3L, 1L))

  # Test perfect prediction
  loss_perfect <- mlx_l1_loss(preds, preds)
  expect_equal(as.numeric(as.matrix(loss_perfect)), 0, tolerance = 1e-6)
})

test_that("binary cross entropy works correctly", {
  # Tests assume MLX is available

  preds <- as_mlx(matrix(c(0.9, 0.2, 0.8, 0.1), 4, 1))
  targets <- as_mlx(matrix(c(1, 0, 1, 0), 4, 1))

  # Test mean reduction
  loss <- mlx_binary_cross_entropy(preds, targets, reduction = "mean")
  expect_s3_class(loss, "mlx")
  expect_equal(length(dim(loss)), 0L) # scalar

  # Loss should be positive
  expect_true(as.numeric(as.matrix(loss)) > 0)

  # Test sum reduction
  loss_sum <- mlx_binary_cross_entropy(preds, targets, reduction = "sum")
  expect_true(as.numeric(as.matrix(loss_sum)) > as.numeric(as.matrix(loss)))

  # Test none reduction
  loss_none <- mlx_binary_cross_entropy(preds, targets, reduction = "none")
  expect_equal(dim(loss_none), c(4L, 1L))

  # Perfect predictions should give very low loss
  perfect_preds <- as_mlx(matrix(c(0.99, 0.01, 0.99, 0.01), 4, 1))
  loss_perfect <- mlx_binary_cross_entropy(perfect_preds, targets)
  expect_true(as.numeric(as.matrix(loss_perfect)) < 0.1)

  # Very wrong predictions should give high loss
  wrong_preds <- as_mlx(matrix(c(0.01, 0.99, 0.01, 0.99), 4, 1))
  loss_wrong <- mlx_binary_cross_entropy(wrong_preds, targets)
  expect_true(as.numeric(as.matrix(loss_wrong)) > as.numeric(as.matrix(loss)))
})

test_that("cross entropy works correctly", {
  # Tests assume MLX is available

  # 3 samples, 4 classes
  set.seed(1)
  logits <- as_mlx(matrix(rnorm(12), 3, 4))
  targets <- as_mlx(c(1, 3, 2))

  # Test mean reduction
  loss <- mlx_cross_entropy(logits, targets, reduction = "mean")
  expect_s3_class(loss, "mlx")
  expect_equal(length(dim(loss)), 0L) # scalar

  # Loss should be positive
  expect_true(as.numeric(as.matrix(loss)) > 0)

  # Test sum reduction
  loss_sum <- mlx_cross_entropy(logits, targets, reduction = "sum")
  expect_true(as.numeric(as.matrix(loss_sum)) > as.numeric(as.matrix(loss)))

  # Test none reduction
  loss_none <- mlx_cross_entropy(logits, targets, reduction = "none")
  expect_equal(dim(loss_none), 3L)

  # Test with very confident correct predictions
  confident_logits <- matrix(0, 3, 4)
  confident_logits[1, 1] <- 10  # Class 1
  confident_logits[2, 3] <- 10  # Class 3
  confident_logits[3, 2] <- 10  # Class 2
  confident_logits <- as_mlx(confident_logits)

  loss_confident <- mlx_cross_entropy(confident_logits, targets)
  expect_true(as.numeric(as.matrix(loss_confident)) < 0.1)
})

test_that("loss functions handle different input types", {
  # Tests assume MLX is available

  # Test with R vectors/matrices (should auto-convert)
  preds <- matrix(c(1.5, 2.5), 2, 1)
  targets <- matrix(c(1, 2), 2, 1)

  loss <- mlx_mse_loss(preds, targets)
  expect_s3_class(loss, "mlx")

  loss <- mlx_l1_loss(preds, targets)
  expect_s3_class(loss, "mlx")

  # Binary cross entropy
  preds_binary <- matrix(c(0.9, 0.1), 2, 1)
  targets_binary <- matrix(c(1, 0), 2, 1)
  loss <- mlx_binary_cross_entropy(preds_binary, targets_binary)
  expect_s3_class(loss, "mlx")
})

test_that("loss functions work with gradient computation", {
  # Tests assume MLX is available

  # Simple loss function using MSE
  loss_fn <- function(w, x, y) {
    preds <- x %*% w
    mlx_mse_loss(preds, y)
  }

  x <- as_mlx(matrix(1:4, 2, 2))
  y <- as_mlx(matrix(c(1, 2), 2, 1))
  w <- as_mlx(matrix(c(0.5, 0.5), 2, 1))

  # Should be able to compute gradients with respect to w
  grads <- mlx_grad(loss_fn, w, x, y, argnums = 1)
  expect_length(grads, 1)
  expect_s3_class(grads[[1]], "mlx")
  expect_equal(dim(grads[[1]]), c(2L, 1L))
})
