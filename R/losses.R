#' Mean squared error loss
#'
#' Computes the mean squared error between predictions and targets.
#'
#' @param predictions Predicted values as an mlx array.
#' @param targets Target values as an mlx array.
#' @param reduction Type of reduction: "mean" (default), "sum", or "none".
#' @return An mlx array containing the loss.
#' @seealso [mlx.nn.losses.mse_loss](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.losses.mse_loss)
#' @export
#' @examples
#' preds <- mlx_matrix(c(1.5, 2.3, 0.8), 3, 1)
#' targets <- mlx_matrix(c(1, 2, 1), 3, 1)
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
#' @seealso [mlx.nn.losses.l1_loss](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.losses.l1_loss)
#' @export
#' @examples
#' preds <- mlx_matrix(c(1.5, 2.3, 0.8), 3, 1)
#' targets <- mlx_matrix(c(1, 2, 1), 3, 1)
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
#' @seealso [mlx.nn.losses.binary_cross_entropy](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.losses.binary_cross_entropy)
#' @export
#' @examples
#' preds <- mlx_matrix(c(0.9, 0.2, 0.8), 3, 1)
#' targets <- mlx_matrix(c(1, 0, 1), 3, 1)
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
#' @details Uses MLX's fused cross-entropy operation on the GPU and a stable
#'   log-sum-exp and gather calculation on the CPU, without constructing one-hot
#'   arrays. Floating-point logits retain their dtype, including float64 on the
#'   CPU. Target indices are validated in R; mlx targets are copied to R for
#'   validation. Gradients with respect to logits use a stable softmax
#'   calculation, including when a target-class logit is `-Inf`.
#'
#' @param logits A matrix of unnormalized predictions (logits), with one row
#'   per sample and one column per class, as an mlx array or R matrix.
#' @param targets Target class indices as an mlx array or integer vector.
#'   Indices are 1-based, with one non-missing index per sample.
#' @param reduction Type of reduction: "mean" (default), "sum", or "none".
#' @return An mlx array containing the loss.
#' @seealso [mlx.nn.losses.cross_entropy](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.losses.cross_entropy)
#' @export
#' @examples
#' # Logits for 3 samples, 4 classes
#' logits <- mlx_matrix(rnorm(12), 3, 4)
#' targets <- as_mlx(c(1, 3, 2))
#' mlx_cross_entropy(logits, targets)
mlx_cross_entropy <- function(logits, targets, reduction = c("mean", "sum", "none")) {
  reduction <- match.arg(reduction)

  logits <- as_mlx(logits)
  shape <- mlx_shape(logits)
  if (length(shape) != 2L || shape[2] == 0L) {
    stop("logits must be a matrix with at least one class.", call. = FALSE)
  }

  # Validate on the host before passing zero-based indices to the native kernel.
  targets_vec <- as.integer(targets)
  if (length(targets_vec) != shape[1]) {
    stop("targets must provide one class index per sample.", call. = FALSE)
  }
  if (anyNA(targets_vec) || any(targets_vec < 1L | targets_vec > shape[2])) {
    stop("targets contain missing class indices or indices outside 1:n_classes.", call. = FALSE)
  }
  targets <- as_mlx(targets_vec - 1L, dtype = "int32")
  loss <- new_mlx(cpp_mlx_cross_entropy(logits$ptr, targets$ptr))

  switch(reduction,
    "mean" = mlx_mean(loss),
    "sum" = mlx_sum(loss),
    "none" = loss
  )
}
