#!/usr/bin/env Rscript
# Run from the repository root. Compare evaluated losses with the former implementation.
repo_root <- rprojroot::find_root(rprojroot::has_file("DESCRIPTION"))
devtools::load_all(repo_root, quiet = TRUE)
source(file.path(repo_root, "inst", "benchmarks", "bench_helpers.R"))

# Baseline from master before the native rewrite.
old_cross_entropy <- function(logits, targets, reduction = c("mean", "sum", "none")) {
  reduction <- match.arg(reduction)

  logits <- as_mlx(logits)
  targets <- as_mlx(targets)

  # Compute log-softmax for numerical stability
  # log_softmax(x) = x - log(sum(exp(x)))
  log_sum_exp <- mlx_logsumexp(logits, axes = 2, drop = FALSE)
  log_probs <- logits - log_sum_exp

  # Get log probability of target class (targets are 1-indexed)
  n_samples <- dim(logits)[1]
  n_classes <- dim(logits)[2]
  targets_vec <- as.integer(targets)
  if (length(targets_vec) != n_samples) {
    stop("targets must provide one class index per sample.", call. = FALSE)
  }
  if (any(targets_vec < 1L | targets_vec > n_classes)) {
    stop("targets contain class indices outside 1:n_classes.", call. = FALSE)
  }

  logits_dtype <- mlx_dtype(logits)
  eye_dtype <- if (identical(logits_dtype, "float64")) "float64" else "float32"
  one_hot <- mlx_eye(n_classes, dtype = eye_dtype)[targets_vec, , drop = FALSE]
  one_hot <- mlx_cast(one_hot, dtype = logits_dtype)

  # Compute negative log likelihood
  loss <- -mlx_sum(log_probs * one_hot, axes = 2)

  switch(reduction,
    "mean" = mlx_mean(loss),
    "sum" = mlx_sum(loss),
    "none" = loss
  )
}

set.seed(1)
for (device in c("cpu", "gpu")) {
  with_device(device, {
    for (n_classes in c(4L, 1024L, 4096L)) {
      logits <- as_mlx(matrix(rnorm(64L * n_classes), 64L, n_classes))
      targets <- sample.int(n_classes, 64L, replace = TRUE)
      force_mlx(logits)
      # Warm up both paths and check their numerical agreement before timing.
      old_loss <- old_cross_entropy(logits, targets)
      new_loss <- mlx_cross_entropy(logits, targets)
      force_mlx(list(old_loss, new_loss))
      stopifnot(isTRUE(all.equal(as.numeric(old_loss), as.numeric(new_loss),
        tolerance = 1e-6)))
      timings <- bench::mark(
        old = force_mlx(old_cross_entropy(logits, targets)),
        native = force_mlx(mlx_cross_entropy(logits, targets)),
        iterations = 30L, check = FALSE, memory = FALSE, filter_gc = FALSE
      )
      cat(sprintf("%s forward, 64 x %d: old %.3f ms, native %.3f ms, %.1fx speedup\n",
        device, n_classes,
        as.numeric(timings$median[1]) * 1000,
        as.numeric(timings$median[2]) * 1000,
        as.numeric(timings$median[1] / timings$median[2])))

      old_loss_fn <- function(x) old_cross_entropy(x, targets)
      new_loss_fn <- function(x) mlx_cross_entropy(x, targets)
      old_grads <- mlx_grad(old_loss_fn, logits, value = TRUE)
      new_grads <- mlx_grad(new_loss_fn, logits, value = TRUE)
      force_mlx(list(old_grads, new_grads))
      stopifnot(isTRUE(all.equal(as.matrix(old_grads$grads[[1]]), as.matrix(new_grads$grads[[1]]),
        tolerance = 1e-6)))
      timings <- bench::mark(
        old = force_mlx(mlx_grad(old_loss_fn, logits, value = TRUE)),
        native = force_mlx(mlx_grad(new_loss_fn, logits, value = TRUE)),
        iterations = 30L, check = FALSE, memory = FALSE, filter_gc = FALSE
      )
      cat(sprintf("%s forward+gradient, 64 x %d: old %.3f ms, native %.3f ms, %.1fx speedup\n",
        device, n_classes,
        as.numeric(timings$median[1]) * 1000,
        as.numeric(timings$median[2]) * 1000,
        as.numeric(timings$median[1] / timings$median[2])))
    }
  })
}
