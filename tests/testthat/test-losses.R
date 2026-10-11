test_that("MSE loss works correctly", {
  # Tests assume MLX is available

  preds <- mlx_matrix(c(1.5, 2.5, 3.5), 3, 1)
  targets <- mlx_matrix(c(1, 2, 3), 3, 1)

  # Test mean reduction
  loss <- mlx_mse_loss(preds, targets, reduction = "mean")
  expect_s3_class(loss, "mlx")
  expect_equal(length(dim(loss)), 0L) # scalar
  expected <- mean((c(1.5, 2.5, 3.5) - c(1, 2, 3))^2)
  expect_equal(as.numeric(loss), expected, tolerance = 1e-6)

  # Test sum reduction
  loss_sum <- mlx_mse_loss(preds, targets, reduction = "sum")
  expect_equal(as.numeric(loss_sum), expected * 3, tolerance = 1e-6)

  # Test none reduction
  loss_none <- mlx_mse_loss(preds, targets, reduction = "none")
  expect_equal(dim(loss_none), c(3L, 1L))
  expected_none <- (c(1.5, 2.5, 3.5) - c(1, 2, 3))^2
  expect_equal(as.numeric(loss_none), expected_none, tolerance = 1e-6)

  # Test perfect prediction
  loss_perfect <- mlx_mse_loss(preds, preds)
  expect_equal(as.numeric(loss_perfect), 0, tolerance = 1e-6)
})

test_that("L1 loss works correctly", {
  # Tests assume MLX is available

  preds <- mlx_matrix(c(1.5, 2.5, 3.5), 3, 1)
  targets <- mlx_matrix(c(1, 2, 3), 3, 1)

  # Test mean reduction
  loss <- mlx_l1_loss(preds, targets, reduction = "mean")
  expect_s3_class(loss, "mlx")
  expected <- mean(abs(c(1.5, 2.5, 3.5) - c(1, 2, 3)))
  expect_equal(as.numeric(loss), expected, tolerance = 1e-6)

  # Test sum reduction
  loss_sum <- mlx_l1_loss(preds, targets, reduction = "sum")
  expect_equal(as.numeric(loss_sum), expected * 3, tolerance = 1e-6)

  # Test none reduction
  loss_none <- mlx_l1_loss(preds, targets, reduction = "none")
  expect_equal(dim(loss_none), c(3L, 1L))

  # Test perfect prediction
  loss_perfect <- mlx_l1_loss(preds, preds)
  expect_equal(as.numeric(loss_perfect), 0, tolerance = 1e-6)
})

test_that("binary cross entropy works correctly", {
  # Tests assume MLX is available

  preds <- mlx_matrix(c(0.9, 0.2, 0.8, 0.1), 4, 1)
  targets <- mlx_matrix(c(1, 0, 1, 0), 4, 1)

  # Test mean reduction
  loss <- mlx_binary_cross_entropy(preds, targets, reduction = "mean")
  expect_s3_class(loss, "mlx")
  expect_equal(length(dim(loss)), 0L) # scalar

  # Loss should be positive
  expect_gt(as.numeric(loss),  0)

  # Test sum reduction
  loss_sum <- mlx_binary_cross_entropy(preds, targets, reduction = "sum")
  expect_gt(as.numeric(loss_sum), as.numeric(loss))

  # Test none reduction
  loss_none <- mlx_binary_cross_entropy(preds, targets, reduction = "none")
  expect_equal(dim(loss_none), c(4L, 1L))

  # Perfect predictions should give very low loss
  perfect_preds <- mlx_matrix(c(0.99, 0.01, 0.99, 0.01), 4, 1)
  loss_perfect <- mlx_binary_cross_entropy(perfect_preds, targets)
  expect_lt(as.numeric(loss_perfect), 0.1)

  # Very wrong predictions should give high loss
  wrong_preds <- mlx_matrix(c(0.01, 0.99, 0.01, 0.99), 4, 1)
  loss_wrong <- mlx_binary_cross_entropy(wrong_preds, targets)
  expect_gt(as.numeric(loss_wrong), as.numeric(loss))
})

test_that("cross entropy works correctly", {
  # 3 samples, 4 classes
  set.seed(1)
  logits <- mlx_matrix(rnorm(12), 3, 4)
  targets <- as_mlx(c(1, 3, 2))

  # Test mean reduction
  loss <- mlx_cross_entropy(logits, targets, reduction = "mean")
  expect_s3_class(loss, "mlx")
  expect_equal(length(dim(loss)), 0L) # scalar

  # Loss should be positive
  expect_true(as.numeric(loss) > 0)

  # Test sum reduction
  loss_sum <- mlx_cross_entropy(logits, targets, reduction = "sum")
  expect_gt(as.numeric(loss_sum), as.numeric(loss))

  # Test none reduction
  loss_none <- mlx_cross_entropy(logits, targets, reduction = "none")
  expect_equal(mlx_shape(loss_none), 3L)

  # Test with very confident correct predictions
  confident_logits <- matrix(0, 3, 4)
  confident_logits[1, 1] <- 10  # Class 1
  confident_logits[2, 3] <- 10  # Class 3
  confident_logits[3, 2] <- 10  # Class 2
  confident_logits <- as_mlx(confident_logits)

  loss_confident <- mlx_cross_entropy(confident_logits, targets)
  expect_lt(as.numeric(loss_confident), 0.1)
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

  x <- mlx_matrix(1:4, 2, 2)
  y <- mlx_matrix(c(1, 2), 2, 1)
  w <- mlx_matrix(c(0.5, 0.5), 2, 1)

  # Should be able to compute gradients with respect to w
  grads <- mlx_grad(loss_fn, w, x, y, argnums = 1)
  expect_length(grads, 1)
  expect_s3_class(grads[[1]], "mlx")
  expect_equal(dim(grads[[1]]), c(2L, 1L))
})

test_that("cross entropy matches stable R calculations on CPU and GPU", {
  logits_r <- matrix(c(2, -Inf, -1, -2, 3, 1), 2, 3, byrow = TRUE)
  targets <- c(1L, 3L)
  shifted <- logits_r - apply(logits_r, 1, max)
  expected <- log(rowSums(exp(shifted))) - shifted[cbind(1:2, targets)]

  for (device in c("cpu", "gpu")) {
    with_device(device, {
      logits <- as_mlx(logits_r)
      for (indices in list(targets, as_mlx(targets))) {
        loss <- mlx_cross_entropy(logits, indices, reduction = "none")
        expect_equal(mlx_shape(loss), 2L)
        expect_equal(as.numeric(loss), expected, tolerance = 1e-6)
        expect_equal(as.numeric(mlx_cross_entropy(logits, indices)), mean(expected),
          tolerance = 1e-6)
        expect_equal(as.numeric(mlx_cross_entropy(logits, indices, "sum")), sum(expected),
          tolerance = 1e-6)
      }
    })
  }
})

test_that("cross entropy gradients match softmax minus target indicators", {
  targets <- c(1L, 3L)
  loss_fn <- function(x) mlx_cross_entropy(x, targets)
  weighted_loss <- function(x) {
    sum(mlx_cross_entropy(x, targets, "none") * as_mlx(c(2, -3)))
  }
  for (logits_r in list(
    matrix(c(2, -Inf, -1, -2, 3, 1), 2, 3, byrow = TRUE),
    matrix(c(-Inf, 0, 0, -2, 3, 1), 2, 3, byrow = TRUE),
    matrix(c(-1e8, 0, 0, -2, 3, 1), 2, 3, byrow = TRUE)
  )) {
    shifted <- logits_r - apply(logits_r, 1, max)
    expected <- exp(shifted) / rowSums(exp(shifted))
    expected[cbind(1:2, targets)] <- expected[cbind(1:2, targets)] - 1
    for (device in c("cpu", "gpu")) {
      with_device(device, {
        logits <- as_mlx(logits_r)
        grads <- mlx_grad(loss_fn, logits)
        expect_equal(as.matrix(grads[[1]]), expected / 2, tolerance = 1e-6)
        grads <- mlx_grad(weighted_loss, logits)
        expect_equal(as.matrix(grads[[1]]), expected * c(2, -3), tolerance = 1e-6)
      })
    }
  }
})

test_that("cross entropy preserves float64 precision and handles large offsets", {
  for (device in c("cpu", "gpu")) {
    with_device(device, {
      logits <- as_mlx(matrix(1e8, 2, 3))
      expect_equal(as.numeric(mlx_cross_entropy(logits, c(1L, 3L))), log(3),
        tolerance = 1e-6)
    })
  }
  with_device("cpu", {
    logits <- as_mlx(matrix(c(0, 1, 2), 1, 3), dtype = "float64")
    loss <- mlx_cross_entropy(logits, 2L)
    expect_identical(mlx_dtype(loss), "float64")
    expect_equal(as.numeric(loss), log(sum(exp(c(-2, -1, 0)))) + 1,
      tolerance = 1e-12)
  })
})

test_that("cross entropy rejects invalid shapes and target indices", {
  logits <- as_mlx(matrix(0, 2, 3))
  expect_error(mlx_cross_entropy(logits, 1L), "one class index per sample")
  for (targets in list(c(0L, 1L), c(1L, 4L), c(1L, NA_integer_))) {
    expect_error(mlx_cross_entropy(logits, targets), "class indices")
  }
  expect_error(mlx_cross_entropy(as_mlx(1:3), 1L), "matrix")
})

test_that("cross entropy handles large class counts and integer logits", {
  for (device in c("cpu", "gpu")) {
    with_device(device, {
      logits <- as_mlx(matrix(0, 2, 100000))
      expect_equal(as.numeric(mlx_cross_entropy(logits, c(1L, 100000L))), log(100000),
        tolerance = 1e-6)
      logits <- as_mlx(matrix(0L, 2, 3), dtype = "int32")
      expect_equal(as.numeric(mlx_cross_entropy(logits, c(1L, 3L))), log(3),
        tolerance = 1e-6)
    })
  }
})


test_that("compiled cross entropy keeps stable gradients", {
  targets <- c(1L, 3L)
  loss_fn <- function(x) mlx_cross_entropy(x, targets)
  logits_r <- matrix(c(-1e8, 0, 0, -2, 3, 1), 2, 3, byrow = TRUE)
  shifted <- logits_r - apply(logits_r, 1, max)
  expected <- exp(shifted) / rowSums(exp(shifted))
  expected[cbind(1:2, targets)] <- expected[cbind(1:2, targets)] - 1
  for (device in c("cpu", "gpu")) {
    with_device(device, {
      compiled_loss <- mlx_compile(loss_fn)
      logits <- as_mlx(logits_r)
      expect_equal(as.numeric(compiled_loss(logits)), as.numeric(loss_fn(logits)),
        tolerance = 1e-6)
      grads <- mlx_grad(compiled_loss, logits)
      expect_equal(as.matrix(grads[[1]]), expected / 2, tolerance = 1e-6)
    })
  }
})
