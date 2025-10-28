#' Mean squared error loss
#'
#' Computes the mean squared error between predictions and targets.
#'
#' @param predictions Predicted values as an mlx array.
#' @param targets Target values as an mlx array.
#' @param reduction Type of reduction: "mean" (default), "sum", or "none".
#' @return An mlx array containing the loss.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.losses.mse_loss}
#' @export
#' @examples
#' preds <- as_mlx(matrix(c(1.5, 2.3, 0.8), 3, 1))
#' targets <- as_mlx(matrix(c(1, 2, 1), 3, 1))
#' mlx_mse_loss(preds, targets)
mlx_mse_loss <- function(predictions, targets, reduction = c("mean", "sum", "none")) {
  reduction <- match.arg(reduction)

  predictions <- as_mlx(predictions)
  targets <- as_mlx(targets)

  squared_diff <- (predictions - targets)^2

  switch(reduction,
    "mean" = mlx_mean(squared_diff),
    "sum" = mlx_sum(squared_diff),
    "none" = squared_diff
  )
}

#' L1 loss (Mean Absolute Error)
#'
#' Computes the mean absolute error between predictions and targets.
#'
#' @param predictions Predicted values as an mlx array.
#' @param targets Target values as an mlx array.
#' @param reduction Type of reduction: "mean" (default), "sum", or "none".
#' @return An mlx array containing the loss.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.losses.l1_loss}
#' @export
#' @examples
#' preds <- as_mlx(matrix(c(1.5, 2.3, 0.8), 3, 1))
#' targets <- as_mlx(matrix(c(1, 2, 1), 3, 1))
#' mlx_l1_loss(preds, targets)
mlx_l1_loss <- function(predictions, targets, reduction = c("mean", "sum", "none")) {
  reduction <- match.arg(reduction)

  predictions <- as_mlx(predictions)
  targets <- as_mlx(targets)

  abs_diff <- abs(predictions - targets)

  switch(reduction,
    "mean" = mlx_mean(abs_diff),
    "sum" = mlx_sum(abs_diff),
    "none" = abs_diff
  )
}

#' Binary cross-entropy loss
#'
#' Computes binary cross-entropy loss between predictions and binary targets.
#'
#' @param predictions Predicted probabilities as an mlx array (values in \[0,1\]).
#' @param targets Binary target values as an mlx array (0 or 1).
#' @param reduction Type of reduction: "mean" (default), "sum", or "none".
#' @return An mlx array containing the loss.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.losses.binary_cross_entropy}
#' @export
#' @examples
#' preds <- as_mlx(matrix(c(0.9, 0.2, 0.8), 3, 1))
#' targets <- as_mlx(matrix(c(1, 0, 1), 3, 1))
#' mlx_binary_cross_entropy(preds, targets)
mlx_binary_cross_entropy <- function(predictions, targets, reduction = c("mean", "sum", "none")) {
  reduction <- match.arg(reduction)

  predictions <- as_mlx(predictions)
  targets <- as_mlx(targets)

  # Clip predictions to avoid log(0)
  eps <- 1e-7
  predictions <- mlx_clip(predictions, min = eps, max = 1 - eps)

  # BCE = -[y*log(p) + (1-y)*log(1-p)]
  loss <- -(targets * log(predictions) + (1 - targets) * log(1 - predictions))

  switch(reduction,
    "mean" = mlx_mean(loss),
    "sum" = mlx_sum(loss),
    "none" = loss
  )
}

#' Cross-entropy loss
#'
#' Computes cross-entropy loss for multi-class classification.
#'
#' @param logits Unnormalized predictions (logits) as an mlx array.
#' @param targets Target class indices as an mlx array or integer vector.
#' @param reduction Type of reduction: "mean" (default), "sum", or "none".
#' @return An mlx array containing the loss.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.losses.cross_entropy}
#' @export
#' @examples
#' # Logits for 3 samples, 4 classes
#' logits <- as_mlx(matrix(rnorm(12), 3, 4))
#' targets <- as_mlx(c(1, 3, 2))  # 0-indexed class labels
#' mlx_cross_entropy(logits, targets)
mlx_cross_entropy <- function(logits, targets, reduction = c("mean", "sum", "none")) {
  reduction <- match.arg(reduction)

  logits <- as_mlx(logits)
  targets <- as_mlx(targets)

  # Compute log-softmax for numerical stability
  # log_softmax(x) = x - log(sum(exp(x)))
  log_sum_exp <- mlx_logsumexp(logits, axis = 2, keepdims = TRUE)
  log_probs <- logits - log_sum_exp

  # Get log probability of target class (targets are assumed 0-indexed)
  n_samples <- logits$dim[1]
  n_classes <- logits$dim[2]
  targets_vec <- as.integer(as.matrix(targets))
  if (length(targets_vec) != n_samples) {
    stop("targets must provide one class index per sample.", call. = FALSE)
  }
  if (any(targets_vec < 0L | targets_vec >= n_classes)) {
    stop("targets contain class indices outside 0:(n_classes - 1).", call. = FALSE)
  }

  eye_dtype <- if (identical(logits$dtype, "float64")) "float64" else "float32"
  one_hot <- mlx_eye(n_classes, dtype = eye_dtype, device = logits$device)[targets_vec + 1L, , drop = FALSE]
  one_hot <- .mlx_cast(one_hot, dtype = logits$dtype, device = logits$device)

  # Compute negative log likelihood
  loss <- -mlx_sum(log_probs * one_hot, axis = 2)

  switch(reduction,
    "mean" = mlx_mean(loss),
    "sum" = mlx_sum(loss),
    "none" = loss
  )
}
