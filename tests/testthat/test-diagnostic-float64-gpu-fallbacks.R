test_that("diagnostic: GPU-tagged float64 operations expose CPU fallbacks", {
  skip_if_not(mlx_has_gpu())

  old_device <- mlx_default_device()
  on.exit(mlx_default_device(old_device), add = TRUE)
  mlx_default_device("gpu")

  force_mlx <- function(x) {
    if (is_mlx(x)) {
      mlx_eval(x)
      return(sprintf("mlx:%s/%s", mlx_dtype(x), mlx_device(x)))
    }
    if (is.list(x)) {
      return(paste(vapply(x, force_mlx, character(1)), collapse = ","))
    }
    sprintf("r:%s", paste(class(x), collapse = "/"))
  }

  classify <- function(name, expr, note = "") {
    result <- tryCatch(
      {
        value <- expr()
        forced <- force_mlx(value)
        list(
          name = name,
          status = "success",
          detail = forced,
          note = note
        )
      },
      error = function(e) {
        message <- conditionMessage(e)
        status <- if (grepl("float64 is not supported on the GPU", message, fixed = TRUE)) {
          "gpu_float64_error"
        } else if (grepl("not yet supported on the GPU", message, fixed = TRUE)) {
          "gpu_unsupported"
        } else if (grepl("Does not yet support given type: float64", message, fixed = TRUE)) {
          "unsupported_float64"
        } else {
          "other_error"
        }
        list(
          name = name,
          status = status,
          detail = message,
          note = note
        )
      }
    )
    as.data.frame(result, stringsAsFactors = FALSE)
  }

  x <- as_mlx(c(0.25, 0.5, 0.75, 1), dtype = "float64", device = "gpu")
  y <- as_mlx(c(1, 2, 3, 4), dtype = "float64", device = "gpu")
  m <- as_mlx(matrix(c(4, 1, 2, 3), 2, 2), dtype = "float64", device = "gpu")
  n <- as_mlx(matrix(c(1, 2, 3, 5), 2, 2), dtype = "float64", device = "gpu")
  cube <- as_mlx(array(seq_len(8), c(2, 2, 2)), dtype = "float64", device = "gpu")
  spd <- as_mlx(matrix(c(4, 1, 1, 3), 2, 2), dtype = "float64", device = "gpu")
  lower <- as_mlx(matrix(c(2, 1, 0, 3), 2, 2), dtype = "float64", device = "gpu")
  cond <- as_mlx(c(TRUE, FALSE, TRUE, FALSE), dtype = "bool", device = "gpu")
  idx <- as_mlx(c(1L, 2L), dtype = "int32", device = "gpu")
  logits <- as_mlx(matrix(c(1, 2, 3, 1, 0, 2), 2, 3), dtype = "float64", device = "gpu")

  cases <- list(
    "%*%.mlx" = function() m %*% n,
    "abind" = function() abind(x, y, along = 1),
    "all.equal.mlx" = function() all.equal(x, y),
    "aperm.mlx" = function() aperm(cube, c(3, 2, 1)),
    "asplit.mlx" = function() asplit(m, 1),
    "cbind.mlx" = function() cbind(m, n),
    "chol.mlx" = function() chol(spd),
    "chol2inv" = function() chol2inv(chol(spd)),
    "colMeans.mlx" = function() colMeans(m),
    "colSums.mlx" = function() colSums(m),
    "crossprod.mlx" = function() crossprod(m, n),
    "diag.mlx" = function() diag(m),
    "drop.mlx" = function() drop(mlx_reshape(x, c(1, 4))),
    "fft.mlx" = function() fft(x),
    "kronecker.mlx" = function() kronecker(m, n),
    "Math abs" = function() abs(x - 0.5),
    "Math acos" = function() acos(x),
    "Math acosh" = function() acosh(y),
    "Math asin" = function() asin(x),
    "Math asinh" = function() asinh(x),
    "Math atan" = function() atan(x),
    "Math atanh" = function() atanh(x / 2),
    "Math ceiling" = function() ceiling(y),
    "Math cos" = function() cos(x),
    "Math cosh" = function() cosh(x),
    "Math cumprod" = function() cumprod(x),
    "Math cumsum" = function() cumsum(x),
    "Math exp" = function() exp(x),
    "Math expm1" = function() expm1(x),
    "Math floor" = function() floor(y),
    "Math log" = function() log(y),
    "Math log10" = function() log10(y),
    "Math log1p" = function() log1p(y),
    "Math log2" = function() log2(y),
    "Math round" = function() round(y),
    "Math sign" = function() sign(x - 0.5),
    "Math sin" = function() sin(x),
    "Math sinh" = function() sinh(x),
    "Math sqrt" = function() sqrt(x),
    "Math tan" = function() tan(x),
    "Math tanh" = function() tanh(x),
    "mean.mlx" = function() mean(m),
    "mlx_addmm" = function() mlx_addmm(m, m, n),
    "mlx_all" = function() mlx_all(x > 0),
    "mlx_allclose" = function() mlx_allclose(x, y, device = "gpu"),
    "mlx_any" = function() mlx_any(x > 0),
    "mlx_argmax" = function() mlx_argmax(x),
    "mlx_argmin" = function() mlx_argmin(x),
    "mlx_argpartition" = function() mlx_argpartition(x, 1),
    "mlx_argsort" = function() mlx_argsort(x),
    "mlx_binary_cross_entropy" = function() mlx_binary_cross_entropy(x, y / 4, reduction = "none"),
    "mlx_broadcast_arrays" = function() {
      mlx_broadcast_arrays(x, mlx_reshape(x, c(1, 4)), device = "gpu")
    },
    "mlx_broadcast_to" = function() mlx_broadcast_to(x, c(2, 4), device = "gpu"),
    "mlx_cast" = function() mlx_cast(x, dtype = "float64", device = "gpu"),
    "mlx_cholesky_inv" = function() mlx_cholesky_inv(chol(spd)),
    "mlx_clip" = function() mlx_clip(x, min = 0.3, max = 0.8),
    "mlx_conjugate" = function() mlx_conjugate(x),
    "mlx_contiguous" = function() mlx_contiguous(m, device = "gpu"),
    "mlx_cross" = function() mlx_cross(as_mlx(matrix(c(1, 0, 0), 1, 3), dtype = "float64", device = "gpu"), as_mlx(matrix(c(0, 1, 0), 1, 3), dtype = "float64", device = "gpu")),
    "mlx_cross_entropy" = function() mlx_cross_entropy(logits, c(1L, 3L), reduction = "none"),
    "mlx_cumprod" = function() mlx_cumprod(m, axis = 1),
    "mlx_cumsum" = function() mlx_cumsum(m, axis = 1),
    "mlx_degrees" = function() mlx_degrees(x),
    "mlx_dexp" = function() mlx_dexp(y, device = "gpu"),
    "mlx_diagonal" = function() mlx_diagonal(m),
    "mlx_dlnorm" = function() mlx_dlnorm(y, device = "gpu"),
    "mlx_dlogis" = function() mlx_dlogis(x, device = "gpu"),
    "mlx_dnorm" = function() mlx_dnorm(x, device = "gpu"),
    "mlx_dunif" = function() mlx_dunif(x, device = "gpu"),
    "mlx_eig" = function() mlx_eig(spd),
    "mlx_eigh" = function() mlx_eigh(spd),
    "mlx_eigvals" = function() mlx_eigvals(spd),
    "mlx_eigvalsh" = function() mlx_eigvalsh(spd),
    "mlx_erf" = function() mlx_erf(x),
    "mlx_erfinv" = function() mlx_erfinv(x / 2),
    "mlx_expand_dims" = function() mlx_expand_dims(x, 1),
    "mlx_fft" = function() mlx_fft(x),
    "mlx_fft2" = function() mlx_fft2(m),
    "mlx_fftn" = function() mlx_fftn(cube),
    "mlx_flatten" = function() mlx_flatten(m),
    "mlx_forward(mlx_batch_norm)" = function() mlx_forward(mlx_batch_norm(2, device = "gpu"), m),
    "mlx_forward(mlx_dropout)" = function() mlx_forward(mlx_dropout(0.5), x),
    "mlx_forward(mlx_gelu)" = function() mlx_forward(mlx_gelu(), x),
    "mlx_forward(mlx_layer_norm)" = function() mlx_forward(mlx_layer_norm(2, device = "gpu"), m),
    "mlx_forward(mlx_leaky_relu)" = function() mlx_forward(mlx_leaky_relu(), x - 0.5),
    "mlx_forward(mlx_relu)" = function() mlx_forward(mlx_relu(), x - 0.5),
    "mlx_forward(mlx_sigmoid)" = function() mlx_forward(mlx_sigmoid(), x),
    "mlx_forward(mlx_silu)" = function() mlx_forward(mlx_silu(), x),
    "mlx_forward(mlx_softmax_layer)" = function() mlx_forward(mlx_softmax_layer(axis = 2), logits),
    "mlx_forward(mlx_tanh)" = function() mlx_forward(mlx_tanh(), x),
    "mlx_gather" = function() mlx_gather(m, list(idx, idx), axes = c(1, 2)),
    "mlx_hadamard_transform" = function() mlx_hadamard_transform(x),
    "mlx_imag" = function() mlx_imag(x),
    "mlx_inv" = function() mlx_inv(spd),
    "mlx_isclose" = function() mlx_isclose(x, y, device = "gpu"),
    "mlx_isfinite" = function() mlx_isfinite(x),
    "mlx_isinf" = function() mlx_isinf(x),
    "mlx_isnan" = function() mlx_isnan(x),
    "mlx_isneginf" = function() mlx_isneginf(x),
    "mlx_isposinf" = function() mlx_isposinf(x),
    "mlx_kron" = function() mlx_kron(m, n),
    "mlx_l1_loss" = function() mlx_l1_loss(x, y, reduction = "none"),
    "mlx_logcumsumexp" = function() mlx_logcumsumexp(logits, axis = 2),
    "mlx_logsumexp" = function() mlx_logsumexp(logits, axes = 2),
    "mlx_lu" = function() mlx_lu(m),
    "mlx_maximum" = function() mlx_maximum(x, y),
    "mlx_mean" = function() mlx_mean(m),
    "mlx_meshgrid" = function() mlx_meshgrid(x, y),
    "mlx_minimum" = function() mlx_minimum(x, y),
    "mlx_moveaxis" = function() mlx_moveaxis(cube, 1, 3),
    "mlx_mse_loss" = function() mlx_mse_loss(x, y, reduction = "none"),
    "mlx_nan_to_num" = function() mlx_nan_to_num(x),
    "mlx_norm" = function() mlx_norm(m),
    "mlx_pad" = function() mlx_pad(m, c(1, 1)),
    "mlx_partition" = function() mlx_partition(x, 1),
    "mlx_pexp" = function() mlx_pexp(y, device = "gpu"),
    "mlx_plnorm" = function() mlx_plnorm(y, device = "gpu"),
    "mlx_plogis" = function() mlx_plogis(x, device = "gpu"),
    "mlx_pnorm" = function() mlx_pnorm(x, device = "gpu"),
    "mlx_prod" = function() mlx_prod(m),
    "mlx_punif" = function() mlx_punif(x, device = "gpu"),
    "mlx_put_along_axis" = function() mlx_put_along_axis(m, as_mlx(matrix(c(1L, 2L), 2, 1), dtype = "int32", device = "gpu"), as_mlx(matrix(c(9, 8), 2, 1), dtype = "float64", device = "gpu"), axis = 2),
    "mlx_qexp" = function() mlx_qexp(x, device = "gpu"),
    "mlx_qlnorm" = function() mlx_qlnorm(x, device = "gpu"),
    "mlx_qlogis" = function() mlx_qlogis(x, device = "gpu"),
    "mlx_qnorm" = function() mlx_qnorm(x, device = "gpu"),
    "mlx_quantile" = function() mlx_quantile(x, 0.5, device = "gpu"),
    "mlx_qunif" = function() mlx_qunif(x, device = "gpu"),
    "mlx_radians" = function() mlx_radians(x),
    "mlx_real" = function() mlx_real(x),
    "mlx_repeat" = function() mlx_repeat(x, 2),
    "mlx_reshape" = function() mlx_reshape(x, c(2, 2)),
    "mlx_roll" = function() mlx_roll(x, 1),
    "mlx_scatter_add_axis" = function() mlx_scatter_add_axis(m, as_mlx(matrix(c(1L, 2L), 2, 1), dtype = "int32", device = "gpu"), as_mlx(matrix(c(9, 8), 2, 1), dtype = "float64", device = "gpu"), axis = 2),
    "mlx_sd" = function() mlx_sd(m),
    "mlx_slice_update" = function() {
      update <- as_mlx(c(9, 9), dtype = "float64", device = "gpu")
      mlx_slice_update(x, update, start = 2, stop = 3)
    },
    "mlx_softmax" = function() mlx_softmax(logits, axes = 2),
    "mlx_solve_triangular" = function() mlx_solve_triangular(lower, x[1:2], upper = FALSE),
    "mlx_sort" = function() mlx_sort(x),
    "mlx_split" = function() mlx_split(m, 2, axis = 1),
    "mlx_squeeze" = function() mlx_squeeze(mlx_reshape(x, c(1, 4, 1))),
    "mlx_stack" = function() mlx_stack(x, y),
    "mlx_std" = function() mlx_std(m),
    "mlx_sum" = function() mlx_sum(m),
    "mlx_sum axis" = function() mlx_sum(m, axes = 1),
    "mlx_swapaxes" = function() mlx_swapaxes(cube, 1, 3),
    "mlx_take_along_axis" = function() mlx_take_along_axis(m, as_mlx(matrix(c(1L, 2L), 2, 1), dtype = "int32", device = "gpu"), axis = 2),
    "mlx_tile" = function() mlx_tile(x, 2),
    "mlx_topk" = function() mlx_topk(x, 2),
    "mlx_trace" = function() mlx_trace(m),
    "mlx_tri_inv" = function() mlx_tri_inv(lower, upper = FALSE),
    "mlx_tril" = function() mlx_tril(m),
    "mlx_triu" = function() mlx_triu(m),
    "mlx_unflatten" = function() mlx_unflatten(x, 1, c(2, 2)),
    "mlx_var" = function() mlx_var(m),
    "mlx_where" = function() mlx_where(cond, x, y),
    "Ops %%" = function() y %% x,
    "Ops %/%" = function() y %/% x,
    "Ops &" = function() (x > 0) & (y > 0),
    "Ops *" = function() x * y,
    "Ops +" = function() x + y,
    "Ops -" = function() y - x,
    "Ops /" = function() y / x,
    "Ops <" = function() x < y,
    "Ops ==" = function() x == y,
    "Ops ^" = function() y ^ 2,
    "Ops unary -" = function() -x,
    "outer.mlx" = function() outer(x, y),
    "pinv" = function() pinv(m),
    "qr.mlx" = function() qr(m),
    "quantile.mlx" = function() quantile(x, 0.5),
    "rbind.mlx" = function() rbind(x, y),
    "rowMeans.mlx" = function() rowMeans(m),
    "rowSums.mlx" = function() rowSums(m),
    "scale.mlx" = function() scale(m),
    "solve.mlx" = function() solve(spd, x[1:2]),
    "Summary max" = function() max(x),
    "Summary min" = function() min(x),
    "Summary prod" = function() prod(x),
    "Summary sum" = function() sum(x),
    "svd" = function() svd(m),
    "svd.mlx" = function() svd.mlx(m),
    "t.mlx" = function() t(m),
    "tcrossprod.mlx" = function() tcrossprod(m, n)
  )

  notes <- c(
    "all.equal.mlx" = "returns R logical after comparison",
    "chol.mlx" = "MLX reports GPU unsupported",
    "chol2inv" = "MLX reports GPU unsupported",
    "mlx_cast" = "same dtype/device cast: likely no-op",
    "mlx_cholesky_inv" = "MLX reports GPU unsupported",
    "mlx_conjugate" = "real input: likely no-op",
    "mlx_eig" = "MLX reports GPU unsupported",
    "mlx_eigh" = "MLX reports GPU unsupported",
    "mlx_eigvals" = "MLX reports GPU unsupported",
    "mlx_eigvalsh" = "MLX reports GPU unsupported",
    "mlx_inv" = "MLX reports GPU unsupported",
    "mlx_lu" = "MLX reports GPU unsupported",
    "mlx_real" = "real input: likely no-op",
    "mlx_solve_triangular" = "MLX reports GPU unsupported",
    "mlx_tri_inv" = "MLX reports GPU unsupported",
    "pinv" = "MLX reports GPU unsupported",
    "qr.mlx" = "MLX reports GPU unsupported",
    "svd" = "MLX reports GPU unsupported",
    "svd.mlx" = "MLX reports GPU unsupported"
  )

  results <- do.call(
    rbind,
    Map(
      function(name, expr) {
        note <- if (name %in% names(notes)) notes[[name]] else ""
        classify(name, expr, note = note)
      },
      names(cases),
      cases
    )
  )

  successes <- results[results$status == "success", ]
  other_errors <- results[results$status == "other_error", ]
  gpu_unsupported <- results[results$status == "gpu_unsupported", ]
  if (nrow(successes)) {
    message(
      "GPU-tagged float64 operations that did not fail:\n",
      paste(capture.output(print(successes, row.names = FALSE)), collapse = "\n")
    )
  }
  if (nrow(other_errors)) {
    message(
      "GPU-tagged float64 operations that failed for another reason:\n",
      paste(capture.output(print(other_errors, row.names = FALSE)), collapse = "\n")
    )
  }
  if (nrow(gpu_unsupported)) {
    message(
      "GPU-tagged float64 operations where MLX reports GPU unsupported:\n",
      paste(capture.output(print(gpu_unsupported, row.names = FALSE)), collapse = "\n")
    )
  }

  expect_equal(
    other_errors$name,
    character(),
    info = "These diagnostic cases failed for reasons unrelated to GPU float64 scheduling."
  )
  expect_equal(
    successes$name,
    c("mlx_cast", "mlx_conjugate", "mlx_real"),
    info = paste(
      "These GPU-tagged float64 operations did not fail because this diagnostic",
      "uses identity-style calls that do not schedule MLX compute for real input."
    )
  )
})
