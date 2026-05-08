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
    "Ops unary -" = function() -x,
    "Ops +" = function() x + y,
    "Ops -" = function() y - x,
    "Ops *" = function() x * y,
    "Ops /" = function() y / x,
    "Ops ^" = function() y ^ 2,
    "Ops ==" = function() x == y,
    "Ops <" = function() x < y,
    "Ops &" = function() (x > 0) & (y > 0),
    "Ops %/%" = function() y %/% x,
    "Ops %%" = function() y %% x,
    "Math abs" = function() abs(x - 0.5),
    "Math sign" = function() sign(x - 0.5),
    "Math sqrt" = function() sqrt(x),
    "Math floor" = function() floor(y),
    "Math ceiling" = function() ceiling(y),
    "Math round" = function() round(y),
    "Math exp" = function() exp(x),
    "Math log" = function() log(y),
    "Math log10" = function() log10(y),
    "Math log2" = function() log2(y),
    "Math log1p" = function() log1p(y),
    "Math expm1" = function() expm1(x),
    "Math sin" = function() sin(x),
    "Math cos" = function() cos(x),
    "Math tan" = function() tan(x),
    "Math asin" = function() asin(x),
    "Math acos" = function() acos(x),
    "Math atan" = function() atan(x),
    "Math sinh" = function() sinh(x),
    "Math cosh" = function() cosh(x),
    "Math tanh" = function() tanh(x),
    "Math asinh" = function() asinh(x),
    "Math acosh" = function() acosh(y),
    "Math atanh" = function() atanh(x / 2),
    "Math cumsum" = function() cumsum(x),
    "Math cumprod" = function() cumprod(x),
    "mlx_degrees" = function() mlx_degrees(x),
    "mlx_radians" = function() mlx_radians(x),
    "mlx_erf" = function() mlx_erf(x),
    "mlx_erfinv" = function() mlx_erfinv(x / 2),
    "mlx_isnan" = function() mlx_isnan(x),
    "mlx_isinf" = function() mlx_isinf(x),
    "mlx_isfinite" = function() mlx_isfinite(x),
    "mlx_isposinf" = function() mlx_isposinf(x),
    "mlx_isneginf" = function() mlx_isneginf(x),
    "mlx_nan_to_num" = function() mlx_nan_to_num(x),
    "mlx_real" = function() mlx_real(x),
    "mlx_imag" = function() mlx_imag(x),
    "mlx_conjugate" = function() mlx_conjugate(x),
    "Summary sum" = function() sum(x),
    "Summary prod" = function() prod(x),
    "Summary min" = function() min(x),
    "Summary max" = function() max(x),
    "mlx_sum" = function() mlx_sum(m),
    "mlx_sum axis" = function() mlx_sum(m, axes = 1),
    "mlx_prod" = function() mlx_prod(m),
    "mlx_mean" = function() mlx_mean(m),
    "mean.mlx" = function() mean(m),
    "mlx_var" = function() mlx_var(m),
    "mlx_std" = function() mlx_std(m),
    "mlx_sd" = function() mlx_sd(m),
    "mlx_all" = function() mlx_all(x > 0),
    "mlx_any" = function() mlx_any(x > 0),
    "mlx_cumsum" = function() mlx_cumsum(m, axis = 1),
    "mlx_cumprod" = function() mlx_cumprod(m, axis = 1),
    "mlx_clip" = function() mlx_clip(x, min = 0.3, max = 0.8),
    "mlx_maximum" = function() mlx_maximum(x, y),
    "mlx_minimum" = function() mlx_minimum(x, y),
    "mlx_softmax" = function() mlx_softmax(logits, axes = 2),
    "mlx_logsumexp" = function() mlx_logsumexp(logits, axes = 2),
    "mlx_logcumsumexp" = function() mlx_logcumsumexp(logits, axis = 2),
    "mlx_isclose" = function() mlx_isclose(x, y, device = "gpu"),
    "mlx_allclose" = function() mlx_allclose(x, y, device = "gpu"),
    "t.mlx" = function() t(m),
    "mlx_cast" = function() mlx_cast(x, dtype = "float64", device = "gpu"),
    "mlx_reshape" = function() mlx_reshape(x, c(2, 2)),
    "mlx_stack" = function() mlx_stack(x, y),
    "mlx_squeeze" = function() mlx_squeeze(mlx_reshape(x, c(1, 4, 1))),
    "mlx_expand_dims" = function() mlx_expand_dims(x, 1),
    "mlx_repeat" = function() mlx_repeat(x, 2),
    "mlx_tile" = function() mlx_tile(x, 2),
    "mlx_pad" = function() mlx_pad(m, c(1, 1)),
    "mlx_roll" = function() mlx_roll(x, 1),
    "mlx_moveaxis" = function() mlx_moveaxis(cube, 1, 3),
    "mlx_contiguous" = function() mlx_contiguous(m, device = "gpu"),
    "mlx_flatten" = function() mlx_flatten(m),
    "mlx_swapaxes" = function() mlx_swapaxes(cube, 1, 3),
    "aperm.mlx" = function() aperm(cube, c(3, 2, 1)),
    "drop.mlx" = function() drop(mlx_reshape(x, c(1, 4))),
    "mlx_split" = function() mlx_split(x, 2),
    "asplit.mlx" = function() asplit(m, 1),
    "mlx_unflatten" = function() mlx_unflatten(x, 1, c(2, 2)),
    "mlx_meshgrid" = function() mlx_meshgrid(x, y),
    "mlx_broadcast_to" = function() mlx_broadcast_to(x, c(2, 4), device = "gpu"),
    "mlx_broadcast_arrays" = function() mlx_broadcast_arrays(x, m, device = "gpu"),
    "mlx_where" = function() mlx_where(cond, x, y),
    "mlx_tril" = function() mlx_tril(m),
    "mlx_triu" = function() mlx_triu(m),
    "diag.mlx" = function() diag(m),
    "mlx_slice_update" = function() mlx_slice_update(x, as_mlx(c(9, 9), dtype = "float64", device = "gpu"), 2),
    "mlx_gather" = function() mlx_gather(m, list(idx, idx), axes = c(1, 2)),
    "mlx_take_along_axis" = function() mlx_take_along_axis(m, as_mlx(matrix(c(1L, 2L), 2, 1), dtype = "int32", device = "gpu"), axis = 2),
    "mlx_put_along_axis" = function() mlx_put_along_axis(m, as_mlx(matrix(c(1L, 2L), 2, 1), dtype = "int32", device = "gpu"), as_mlx(matrix(c(9, 8), 2, 1), dtype = "float64", device = "gpu"), axis = 2),
    "mlx_scatter_add_axis" = function() mlx_scatter_add_axis(m, as_mlx(matrix(c(1L, 2L), 2, 1), dtype = "int32", device = "gpu"), as_mlx(matrix(c(9, 8), 2, 1), dtype = "float64", device = "gpu"), axis = 2),
    "abind" = function() abind(x, y, along = 1),
    "rbind.mlx" = function() rbind(x, y),
    "cbind.mlx" = function() cbind(x, y),
    "mlx_sort" = function() mlx_sort(x),
    "mlx_argsort" = function() mlx_argsort(x),
    "mlx_topk" = function() mlx_topk(x, 2),
    "mlx_argmax" = function() mlx_argmax(x),
    "mlx_argmin" = function() mlx_argmin(x),
    "mlx_partition" = function() mlx_partition(x, 1),
    "mlx_argpartition" = function() mlx_argpartition(x, 1),
    "%*%.mlx" = function() m %*% n,
    "mlx_addmm" = function() mlx_addmm(m, m, n),
    "crossprod.mlx" = function() crossprod(m, n),
    "tcrossprod.mlx" = function() tcrossprod(m, n),
    "outer.mlx" = function() outer(x, y),
    "chol.mlx" = function() chol(spd),
    "chol2inv" = function() chol2inv(chol(spd)),
    "solve.mlx" = function() solve(spd, x[1:2]),
    "pinv" = function() pinv(m),
    "mlx_kron" = function() mlx_kron(m, n),
    "kronecker.mlx" = function() kronecker(m, n),
    "mlx_inv" = function() mlx_inv(spd),
    "mlx_tri_inv" = function() mlx_tri_inv(lower, upper = FALSE),
    "mlx_cholesky_inv" = function() mlx_cholesky_inv(chol(spd)),
    "mlx_lu" = function() mlx_lu(m),
    "mlx_norm" = function() mlx_norm(m),
    "mlx_solve_triangular" = function() mlx_solve_triangular(lower, x[1:2], upper = FALSE),
    "mlx_trace" = function() mlx_trace(m),
    "mlx_diagonal" = function() mlx_diagonal(m),
    "mlx_eig" = function() mlx_eig(spd),
    "mlx_eigh" = function() mlx_eigh(spd),
    "mlx_eigvals" = function() mlx_eigvals(spd),
    "mlx_eigvalsh" = function() mlx_eigvalsh(spd),
    "mlx_cross" = function() mlx_cross(as_mlx(matrix(c(1, 0, 0), 1, 3), dtype = "float64", device = "gpu"), as_mlx(matrix(c(0, 1, 0), 1, 3), dtype = "float64", device = "gpu")),
    "mlx_dnorm" = function() mlx_dnorm(x, device = "gpu"),
    "mlx_pnorm" = function() mlx_pnorm(x, device = "gpu"),
    "mlx_qnorm" = function() mlx_qnorm(x, device = "gpu"),
    "mlx_dunif" = function() mlx_dunif(x, device = "gpu"),
    "mlx_punif" = function() mlx_punif(x, device = "gpu"),
    "mlx_qunif" = function() mlx_qunif(x, device = "gpu"),
    "mlx_dexp" = function() mlx_dexp(y, device = "gpu"),
    "mlx_pexp" = function() mlx_pexp(y, device = "gpu"),
    "mlx_qexp" = function() mlx_qexp(x, device = "gpu"),
    "mlx_dlnorm" = function() mlx_dlnorm(y, device = "gpu"),
    "mlx_plnorm" = function() mlx_plnorm(y, device = "gpu"),
    "mlx_qlnorm" = function() mlx_qlnorm(x, device = "gpu"),
    "mlx_dlogis" = function() mlx_dlogis(x, device = "gpu"),
    "mlx_plogis" = function() mlx_plogis(x, device = "gpu"),
    "mlx_qlogis" = function() mlx_qlogis(x, device = "gpu"),
    "mlx_mse_loss" = function() mlx_mse_loss(x, y, reduction = "none"),
    "mlx_l1_loss" = function() mlx_l1_loss(x, y, reduction = "none"),
    "mlx_binary_cross_entropy" = function() mlx_binary_cross_entropy(x, y / 4, reduction = "none"),
    "mlx_cross_entropy" = function() mlx_cross_entropy(logits, c(1L, 3L), reduction = "none"),
    "mlx_forward(mlx_relu)" = function() mlx_forward(mlx_relu(), x - 0.5),
    "mlx_forward(mlx_gelu)" = function() mlx_forward(mlx_gelu(), x),
    "mlx_forward(mlx_sigmoid)" = function() mlx_forward(mlx_sigmoid(), x),
    "mlx_forward(mlx_tanh)" = function() mlx_forward(mlx_tanh(), x),
    "mlx_forward(mlx_silu)" = function() mlx_forward(mlx_silu(), x),
    "mlx_forward(mlx_leaky_relu)" = function() mlx_forward(mlx_leaky_relu(), x - 0.5),
    "mlx_forward(mlx_softmax_layer)" = function() mlx_forward(mlx_softmax_layer(axis = 2), logits),
    "mlx_forward(mlx_dropout)" = function() mlx_forward(mlx_dropout(0.5), x),
    "mlx_forward(mlx_layer_norm)" = function() mlx_forward(mlx_layer_norm(2, device = "gpu"), m),
    "mlx_forward(mlx_batch_norm)" = function() mlx_forward(mlx_batch_norm(2, device = "gpu"), m)
  )

  notes <- c(
    "mlx_real" = "real input: likely no-op",
    "mlx_conjugate" = "real input: likely no-op",
    "Summary sum" = "explicit CpuDefaultDeviceGuard; not individually documented",
    "Summary prod" = "explicit CpuDefaultDeviceGuard; not individually documented",
    "Summary min" = "explicit CpuDefaultDeviceGuard; not individually documented",
    "Summary max" = "explicit CpuDefaultDeviceGuard; not individually documented",
    "mlx_sum" = "explicit CpuDefaultDeviceGuard; not individually documented",
    "mlx_sum axis" = "explicit CpuDefaultDeviceGuard; not individually documented",
    "mlx_prod" = "explicit CpuDefaultDeviceGuard; not individually documented",
    "mlx_mean" = "explicit CpuDefaultDeviceGuard; not individually documented",
    "mean.mlx" = "explicit CpuDefaultDeviceGuard; not individually documented",
    "mlx_var" = "explicit CpuDefaultDeviceGuard; not individually documented",
    "mlx_std" = "explicit CpuDefaultDeviceGuard; not individually documented",
    "mlx_sd" = "explicit CpuDefaultDeviceGuard; not individually documented",
    "mlx_cast" = "same dtype/device cast: likely no-op",
    "mlx_unflatten" = "explicit CPU stream; not documented",
    "diag.mlx" = "explicit CPU stream; not documented",
    "outer.mlx" = "explicit CPU stream; not documented",
    "chol.mlx" = "explicit CPU stream; not documented",
    "chol2inv" = "explicit CPU stream; not documented",
    "pinv" = "explicit CPU stream; not documented",
    "mlx_inv" = "explicit CPU stream; documented as CPU",
    "mlx_tri_inv" = "explicit CPU stream; not documented",
    "mlx_cholesky_inv" = "explicit CPU stream; not documented",
    "mlx_lu" = "explicit CPU stream; not documented",
    "mlx_norm" = "explicit CPU stream; not documented",
    "mlx_trace" = "explicit CPU stream; not documented",
    "mlx_diagonal" = "explicit CPU stream; not documented",
    "mlx_eig" = "explicit CPU stream; not documented",
    "mlx_eigh" = "explicit CPU stream; not documented",
    "mlx_eigvals" = "explicit CPU stream; not documented",
    "mlx_eigvalsh" = "explicit CPU stream; not documented"
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
  if (nrow(successes)) {
    message(
      "GPU-tagged float64 operations that did not fail:\n",
      paste(capture.output(print(successes, row.names = FALSE)), collapse = "\n")
    )
  }

  expect_equal(
    successes$name,
    character(),
    info = paste(
      "These GPU-tagged float64 operations did not fail; investigate whether",
      "they are documented CPU fallbacks, metadata-only operations, or silent CPU moves."
    )
  )
})
