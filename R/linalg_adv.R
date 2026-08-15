#' Cholesky decomposition for mlx arrays
#'
#' If `x` is not symmetric positive semi-definite, "behaviour is undefined"
#' according to the MLX documentation.
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_matrix_required
#' @param pivot Ignored; pivoted decomposition is not supported.
#' @inheritParams ellipsis_ignored
#' @inheritParams common_params
#' @return Upper-triangular Cholesky factor as an mlx matrix.
#' @seealso [mlx.linalg.cholesky](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.cholesky)
#' @export
#' @examples
#' x <- mlx_matrix(c(4, 1, 1, 3), 2, 2)
#' chol(x, device = "cpu")
chol.mlx <- function(x, pivot = FALSE, ..., device = NULL) {
  x <- as_mlx(x)
  if (pivot) stop("pivoted Cholesky is not supported for mlx objects.", call. = FALSE)
  x_dtype <- mlx_dtype(x)
  if (!is.null(device)) local_device(device)
  ptr <- cpp_mlx_cholesky(x$ptr, TRUE, x_dtype)
  new_mlx(ptr, dimnames = dimnames(x))
}

#' Inverse from Cholesky decomposition
#'
#' Compute the inverse of a symmetric, positive definite matrix from its
#' Cholesky decomposition. The input `x` should be an upper triangular matrix
#' from `chol()`.
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_matrix_required
#' @param size Ignored; included for compatibility with base R.
#' @inheritParams ellipsis_ignored
#' @inheritParams common_params
#' @return The inverse of the original matrix (before Cholesky decomposition).
#' @seealso [chol()], [solve()], [mlx_cholesky_inv()]
#' @export
#' @examples
#' A <- mlx_matrix(c(4, 1, 1, 3), 2, 2)
#' U <- chol(A, device = "cpu")
#' A_inv <- chol2inv(U, device = "cpu")
#' # Verify: A %*% A_inv should be identity
#' A %*% A_inv
chol2inv <- function(x, size = NCOL(x), ...) {
  UseMethod("chol2inv")
}

#' @export
#' @rdname chol2inv
chol2inv.default <- function(x, size = NCOL(x), ...) {
  # Call base R's chol2inv
  base::chol2inv(x, size = size, ...)
}

#' @export
#' @rdname chol2inv
chol2inv.mlx <- function(x, size = NCOL(x), ..., device = NULL) {
  x <- as_mlx(x)
  # R's chol() always returns upper triangular, so we always use upper=TRUE
  mlx_cholesky_inv(x, upper = TRUE, device = device)
}

#' QR decomposition for mlx arrays
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_matrix_required
#' @param tol Ignored; custom tolerances are not supported.
#' @param LAPACK Ignored; set to `FALSE`.
#' @inheritParams ellipsis_ignored
#' @inheritParams common_params
#' @return A list with components `Q` and `R`, each an mlx matrix.
#' @seealso [mlx.linalg.qr](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.qr)
#' @export
#' @examples
#' x <- mlx_matrix(c(1, 2, 3, 4, 5, 6), 3, 2)
#' qr(x, device = "cpu")
qr.mlx <- function(x, tol = 1e-7, LAPACK = FALSE, ..., device = NULL) {
  x <- as_mlx(x)
  stopifnot(length(dim(x)) == 2L)
  x_dtype <- mlx_dtype(x)

  if (!missing(tol) && !isTRUE(all.equal(tol, 1e-7))) {
    stop("Custom tolerance is not supported for mlx QR decomposition.", call. = FALSE)
  }
  if (LAPACK) stop("LAPACK = TRUE is not supported for mlx objects.", call. = FALSE)
  if (!is.null(device)) local_device(device)
  res <- cpp_mlx_qr(x$ptr, x_dtype)
  structure(
    list(
      Q = new_mlx(res$Q, dimnames = list(rownames(x), colnames(x))),
      R = new_mlx(res$R, dimnames = list(colnames(x), colnames(x)))
    ),
    class = c("mlx_qr", "list")
  )
}

.mlx_qr_gpu_cache <- new.env(parent = emptyenv())

#' GPU QR reduction for tall least-squares problems
#'
#' Computes the QR quantities needed by large least-squares fits without
#' materializing the full `Q` matrix. This is intended for tall, full-rank
#' real-valued matrices. It returns the final upper-triangular `R` and, when
#' `y` is supplied, `qty = Q' y`.
#'
#' The default method uses a Cholesky QR reduction: it computes `crossprod(x)`
#' and `crossprod(x, y)` on the GPU, then uses MLX linalg on CPU for the small
#' `p` by `p` Cholesky and triangular solve. `method = "cholqr2"` applies a
#' second Cholesky QR pass to improve numerical stability, with its large
#' matrix products on the GPU. `method = "metal_householder"` uses cached
#' custom Metal kernels for unpivoted Householder QR without materializing full
#' `Q`. `method = "blocked_householder"` compiles compact WY
#' Householder panels with MLX and applies each panel with GPU matrix operations.
#' `method = "householder"` uses unblocked Householder updates. `method =
#' "tsqr"` uses custom Metal kernels for a tiled Householder reduction followed
#' by a tree reduction of the small triangular factors. The QR-based paths are
#' more numerically stable but currently slower than Cholesky QR for
#' well-conditioned large problems.
#'
#' GPU work is currently restricted to `float32`. Integer inputs are cast to
#' `float32`; `float64` and complex inputs are not supported on the GPU path.
#'
#' `method = "tsqr"` stores one input tile in Metal threadgroup memory. When
#' `block_rows = NULL`, the tile height is chosen from `p` and `ncol(y)` to fit
#' the 32 KB threadgroup-memory limit and provide enough independent blocks to
#' occupy the GPU.
#'
#' CholeskyQR2 checks the orthogonality of its first pass. If that pass is
#' unsafe, it falls back to GPU TSQR when its compact state fits in threadgroup
#' memory, and otherwise to MLX QR on the CPU.
#'
#' @inheritParams mlx_matrix_required
#' @param y Optional response vector or matrix with `nrow(x)` rows.
#' @param block_rows Number of rows reduced by each first-level GPU block for
#'   `method = "tsqr"`; reduction chunk size for `method = "metal_householder"`.
#'   The default `NULL` chooses a GPU tile size automatically for TSQR and uses
#'   2048 rows for Metal Householder.
#' @param tol Relative tolerance for detecting rank deficiency from `diag(R)`.
#' @param method `"cholqr"` for the fast default Cholesky QR path,
#'   `"cholqr2"` for a second, stabilizing Cholesky QR pass,
#'   `"metal_householder"` for custom Metal Householder QR,
#'   `"blocked_householder"` for compact WY Householder QR using MLX GPU
#'   matrix operations, `"householder"` for direct Householder QR using MLX GPU
#'   primitives, or `"tsqr"` for the custom Metal tall-skinny QR reduction.
#' @return A list with components `R`, optional `qty`, `rank`, `pivot`, and
#'   `block_rows`.
#' @export
#' @examples
#' \dontrun{
#' x <- as_mlx(matrix(rnorm(1000 * 8), 1000, 8))
#' y <- as_mlx(matrix(rnorm(1000), 1000, 1))
#' fit <- mlx_qr_gpu(x, y)
#' coef <- mlx_solve_triangular(fit$R, fit$qty, upper = TRUE, device = "cpu")
#' }
mlx_qr_gpu <- function(x,
                       y = NULL,
                       block_rows = NULL,
                       tol = 1e-4,
                       method = c("cholqr", "cholqr2", "metal_householder",
                                  "blocked_householder",
                                  "householder", "tsqr")) {
  method <- match.arg(method)
  if (!mlx_has_gpu()) {
    stop("mlx_qr_gpu() requires an MLX GPU device.", call. = FALSE)
  }
  local_device("gpu")

  x <- as_mlx(x)
  x_shape <- mlx_shape(x)
  if (length(x_shape) != 2L) {
    stop("mlx_qr_gpu() requires a 2D matrix input.", call. = FALSE)
  }

  n <- as.integer(x_shape[[1L]])
  p <- as.integer(x_shape[[2L]])
  automatic_block_rows <- is.null(block_rows)
  if (!automatic_block_rows) {
    block_rows <- as.integer(block_rows[[1L]])
    if (is.na(block_rows) || block_rows < 1L) {
      stop("block_rows must be NULL or a positive integer.", call. = FALSE)
    }
  }
  if (p < 1L || n < 1L) {
    stop("mlx_qr_gpu() requires a non-empty matrix.", call. = FALSE)
  }
  uses_custom_tsqr <- identical(method, "tsqr")

  x_dtype <- mlx_dtype(x)
  if (identical(x_dtype, "float64")) {
    stop("mlx_qr_gpu() does not support float64 on the GPU path.", call. = FALSE)
  }
  if (identical(x_dtype, "complex64")) {
    stop("mlx_qr_gpu() does not support complex inputs.", call. = FALSE)
  }
  if (!identical(x_dtype, "float32")) {
    x <- mlx_cast(x, "float32")
  }

  has_y <- !is.null(y)
  if (has_y) {
    y <- as_mlx(y)
    y_dtype <- mlx_dtype(y)
    if (identical(y_dtype, "float64")) {
      stop("mlx_qr_gpu() does not support float64 responses on the GPU path.",
           call. = FALSE)
    }
    if (identical(y_dtype, "complex64")) {
      stop("mlx_qr_gpu() does not support complex responses.", call. = FALSE)
    }
    if (!identical(y_dtype, "float32")) {
      y <- mlx_cast(y, "float32")
    }

    y_shape <- mlx_shape(y)
    if (length(y_shape) == 1L) {
      if (as.integer(y_shape[[1L]]) != n) {
        stop("y must have length nrow(x).", call. = FALSE)
      }
      y <- mlx_reshape(y, c(n, 1L))
      y_cols <- 1L
    } else if (length(y_shape) == 2L) {
      if (as.integer(y_shape[[1L]]) != n) {
        stop("y must have nrow(x) rows.", call. = FALSE)
      }
      y_cols <- as.integer(y_shape[[2L]])
    } else {
      stop("y must be a vector or 2D matrix.", call. = FALSE)
    }
  } else {
    y <- mlx_zeros(c(n, 1L), dtype = "float32")
    y_cols <- 1L
  }

  if (automatic_block_rows) {
    if (uses_custom_tsqr) {
      max_tile_rows <- as.integer(floor(
        (32768L - 4L * (p + y_cols + 16L)) / (4L * (p + y_cols))
      ))
      block_rows <- min(256L, max_tile_rows)
    } else {
      block_rows <- 2048L
    }
  }
  if (uses_custom_tsqr && y_cols > 64L) {
    stop(
      "mlx_qr_gpu(method = \"tsqr\") currently supports at most 64 response columns ",
      "because the custom Metal kernels store a full p by ncol(y) working matrix ",
      "in threadgroup memory.",
      call. = FALSE
    )
  }
  if (uses_custom_tsqr) {
    combine_bytes <- 4L * (p * p + p * y_cols + p + y_cols + 3L)
    tile_bytes <- 4L * (
      block_rows * (p + y_cols) + p + y_cols + 16L
    )
    if (block_rows < 1L || combine_bytes > 32768L || tile_bytes > 32768L) {
      stop(
        "mlx_qr_gpu(method = \"tsqr\") exceeds this GPU's 32 KB threadgroup ",
        "memory limit for the requested columns, responses, or block_rows.",
        call. = FALSE
      )
    }
  }

  if (method %in% c("cholqr", "cholqr2")) {
    stable_qr_fallback <- function() {
      if (4L * (p * p + p * y_cols + p + y_cols + 3L) <= 32768L) {
        fallback <- mlx_qr_gpu(
          x, if (has_y) y else NULL,
          block_rows = if (automatic_block_rows) NULL else block_rows,
          tol = tol, method = "tsqr"
        )
        fallback$requested_method <- "cholqr2"
        return(fallback)
      }

      cpu_fit <- qr(x, device = "cpu")
      cpu_diag <- abs(as.numeric(diag(cpu_fit$R)))
      cpu_rank_tol <- tol * max(1, max(cpu_diag, na.rm = TRUE))
      cpu_rank <- sum(is.finite(cpu_diag) & cpu_diag > cpu_rank_tol)
      if (cpu_rank < p) {
        stop(sprintf(
          "mlx_qr_gpu() detected rank deficiency: rank %d < %d.",
          cpu_rank, p
        ), call. = FALSE)
      }
      fallback <- list(
        R = cpu_fit$R,
        rank = cpu_rank,
        pivot = seq_len(p),
        block_rows = block_rows,
        method = "cpu_qr",
        requested_method = "cholqr2"
      )
      if (has_y) {
        fallback$qty <- with_device("cpu", crossprod(cpu_fit$Q, y))
      }
      structure(fallback, class = c("mlx_qr_gpu", "list"))
    }

    xtx <- crossprod(x)
    r_first <- tryCatch(
      chol(xtx, device = "cpu"),
      error = function(e) {
        if (identical(method, "cholqr2")) {
          return(NULL)
        }
        stop("mlx_qr_gpu() detected rank deficiency: Cholesky QR failed.",
             call. = FALSE)
      }
    )

    if (is.null(r_first)) {
      return(stable_qr_fallback())
    }

    diag_vals <- abs(as.numeric(diag(r_first)))
    rank_tol <- tol * max(1, max(diag_vals, na.rm = TRUE))
    rank <- sum(is.finite(diag_vals) & diag_vals > rank_tol)
    if (rank < p) {
      if (identical(method, "cholqr2")) {
        return(stable_qr_fallback())
      }
      stop(sprintf(
        "mlx_qr_gpu() detected rank deficiency: rank %d < %d.",
        rank, p
      ), call. = FALSE)
    }

    if (identical(method, "cholqr")) {
      out <- list(
        R = r_first,
        rank = rank,
        pivot = seq_len(p),
        block_rows = block_rows,
        method = method
      )
      if (has_y) {
        xty <- crossprod(x, y)
        out$qty <- mlx_solve_triangular(t(r_first), xty,
                                        upper = FALSE, device = "cpu")
      }
      return(structure(out, class = c("mlx_qr_gpu", "list")))
    }

    r_first_inv <- mlx_solve_triangular(
      r_first, mlx_eye(p), upper = TRUE, device = "cpu"
    )
    q_first <- x %*% r_first_inv
    q_first_gram <- crossprod(q_first)
    orthogonality_error <- as.numeric(max(abs(q_first_gram - mlx_eye(p))))

    if (!is.finite(orthogonality_error) || orthogonality_error > 0.2) {
      return(stable_qr_fallback())
    }

    r_second <- tryCatch(
      chol(q_first_gram, device = "cpu"),
      error = function(e) NULL
    )
    if (is.null(r_second)) {
      return(stable_qr_fallback())
    }

    r_final <- r_second %*% r_first
    out <- list(
      R = r_final,
      rank = rank,
      pivot = seq_len(p),
      block_rows = block_rows,
      method = method,
      orthogonality_error = orthogonality_error
    )
    if (has_y) {
      q_first_ty <- crossprod(q_first, y)
      out$qty <- mlx_solve_triangular(
        t(r_second), q_first_ty, upper = FALSE, device = "cpu"
      )
    }
    return(structure(out, class = c("mlx_qr_gpu", "list")))
  }

  if (identical(method, "metal_householder")) {
    if (is.null(.mlx_qr_gpu_cache$hh_norm_kernel)) {
      .mlx_qr_gpu_cache$hh_norm_kernel <- mlx_metal_kernel(
        name = "rmlx_qr_gpu_hh_norm",
        input_names = "x",
        output_names = "partial",
        ensure_row_contiguous = TRUE,
        source = "
          // First phase of one Householder column: each threadgroup reduces a
          // contiguous chunk of column K and emits one partial squared norm.
          // A later MLX sum reduces these partials to the scalar norm.
          uint group = threadgroup_position_in_grid.x;
          uint lid = thread_position_in_threadgroup.x;
          int n = x_shape[0];
          int row_start = K + group * REDUCE_ROWS;
          int row_stop = metal::min(n, row_start + REDUCE_ROWS);

          float local_sum = 0.0f;
          for (int row = row_start + lid; row < row_stop; row += THREADS) {
            float val = (float)x[row * P + K];
            local_sum += val * val;
          }

          float sum = simd_sum(local_sum);
          if (lid == 0) {
            partial[group] = sum;
          }
        "
      )
    }

    if (is.null(.mlx_qr_gpu_cache$hh_dot_kernel)) {
      .mlx_qr_gpu_cache$hh_dot_kernel <- mlx_metal_kernel(
        name = "rmlx_qr_gpu_hh_dot",
        input_names = c("x", "y", "params"),
        output_names = "partial",
        ensure_row_contiguous = TRUE,
        source = "
          // Second phase of one Householder column: compute partial dot
          // products v' A[, K:P] and v' y. One threadgroup handles one
          // (row chunk, target column) pair. This keeps the reduction wide
          // enough for the GPU and avoids building full Householder vectors.
          uint packed_group = threadgroup_position_in_grid.x;
          uint chunk = packed_group / TARGET_COLS;
          uint target = packed_group - chunk * TARGET_COLS;
          uint lid = thread_position_in_threadgroup.x;

          int n = x_shape[0];
          int row_start = K + chunk * REDUCE_ROWS;
          int row_stop = metal::min(n, row_start + REDUCE_ROWS);
          float v0 = (float)params[0];

          float local_sum = 0.0f;
          for (int row = row_start + lid; row < row_stop; row += THREADS) {
            float v = row == K ? v0 : (float)x[row * P + K];
            float val;
            if (target < TARGET_X_COLS) {
              val = (float)x[row * P + (K + target)];
            } else {
              int y_col = target - TARGET_X_COLS;
              val = (float)y[row * YCOLS + y_col];
            }
            local_sum += v * val;
          }

          float sum = simd_sum(local_sum);
          if (lid == 0) {
            partial[chunk * TARGET_COLS + target] = sum;
          }
        "
      )
    }

    if (is.null(.mlx_qr_gpu_cache$hh_params_kernel)) {
      .mlx_qr_gpu_cache$hh_params_kernel <- mlx_metal_kernel(
        name = "rmlx_qr_gpu_hh_params",
        input_names = c("x", "partial"),
        output_names = "params",
        ensure_row_contiguous = TRUE,
        source = "
          // Compute Householder parameters entirely on GPU to avoid a
          // per-column CPU scalar synchronization. The output is:
          //   params[0] = v0 = alpha - beta
          //   params[1] = tau = 2 / (v'v)
          //   params[2] = beta, the new diagonal entry before sign fixing
          uint lid = thread_position_in_threadgroup.x;
          int groups = partial_shape[0];

          float local_sum = 0.0f;
          for (int i = lid; i < groups; i += THREADS) {
            local_sum += (float)partial[i];
          }
          float norm_sq = simd_sum(local_sum);

          if (lid == 0) {
            float alpha = (float)x[0];
            float norm_x = metal::sqrt(norm_sq);
            float beta = alpha >= 0.0f ? -norm_x : norm_x;
            float v0 = alpha - beta;
            float tail_sq = metal::max(norm_sq - alpha * alpha, 0.0f);
            float u_norm_sq = v0 * v0 + tail_sq;
            float tau = u_norm_sq == 0.0f ? 0.0f : 2.0f / u_norm_sq;
            params[0] = v0;
            params[1] = tau;
            params[2] = beta;
          }
        "
      )
    }

    if (is.null(.mlx_qr_gpu_cache$hh_update_kernel)) {
      .mlx_qr_gpu_cache$hh_update_kernel <- mlx_metal_kernel(
        name = "rmlx_qr_gpu_hh_update",
        input_names = c("x", "y", "params", "dots"),
        output_names = c("x_out", "y_out"),
        ensure_row_contiguous = TRUE,
        source = "
          // Final phase of one Householder column. The pre-reduced dot vector
          // contains v' A[, K:P] followed by v' y. This kernel fuses the
          // trailing matrix update and the matching response update, producing
          // fresh MLX arrays for the next QR step.
          uint elem = thread_position_in_grid.x;
          int n = x_shape[0];
          int total_x = n * P;
          int total_y = n * YCOLS;
          float v0 = (float)params[0];
          float tau = (float)params[1];
          float beta = (float)params[2];

          if (elem < total_x) {
            int row = elem / P;
            int col = elem - row * P;
            float val = (float)x[elem];
            if (row >= K && col >= K && tau != 0.0f) {
              float v = row == K ? v0 : (float)x[row * P + K];
              val -= tau * v * (float)dots[col - K];
              if (col == K) {
                val = row == K ? beta : 0.0f;
              }
            }
            x_out[elem] = val;
          }

          if (elem < total_y) {
            int row = elem / YCOLS;
            int col = elem - row * YCOLS;
            float val = (float)y[elem];
            if (row >= K && tau != 0.0f) {
              float v = row == K ? v0 : (float)x[row * P + K];
              val -= tau * v * (float)dots[TARGET_X_COLS + col];
            }
            y_out[elem] = val;
          }
        "
      )
    }

    if (is.null(.mlx_qr_gpu_cache$hh_compact_update_kernel)) {
      .mlx_qr_gpu_cache$hh_compact_update_kernel <- mlx_metal_kernel(
        name = "rmlx_qr_gpu_hh_compact_update",
        input_names = c("x", "y", "params", "dots"),
        output_names = c("x_next", "r_row", "y_next", "qty_row"),
        ensure_row_contiguous = TRUE,
        source = "
          // Compact Householder update. Instead of writing a full n by p
          // matrix after column 0 is eliminated, this kernel emits:
          //   * the completed R row,
          //   * the completed Q'y row,
          //   * the next trailing matrix A[2:n, 2:p],
          //   * the next trailing response block y[2:n, ].
          // The R wrapper stores the small rows and feeds only the trailing
          // arrays into the next Householder step.
          uint elem = thread_position_in_grid.x;
          int n = x_shape[0];
          int p_cur = x_shape[1];
          int n_next = metal::max(n - 1, 1);
          int p_next = metal::max(p_cur - 1, 1);
          int total_x_next = n_next * p_next;
          int total_y_next = n_next * YCOLS;
          float v0 = (float)params[0];
          float tau = (float)params[1];
          float beta = (float)params[2];

          if (elem < p_cur) {
            float val = (float)x[elem];
            if (tau != 0.0f) {
              val -= tau * v0 * (float)dots[elem];
            }
            if (elem == 0) {
              val = beta;
            }
            r_row[elem] = val;
          }

          if (elem < YCOLS) {
            float val = (float)y[elem];
            if (tau != 0.0f) {
              val -= tau * v0 * (float)dots[p_cur + elem];
            }
            qty_row[elem] = val;
          }

          if (elem < total_x_next) {
            int row_next = elem / p_next;
            int col_next = elem - row_next * p_next;
            if (p_cur == 1) {
              x_next[elem] = 0.0f;
            } else {
              int row = row_next + 1;
              int col = col_next + 1;
              float v = (float)x[row * p_cur];
              float val = (float)x[row * p_cur + col];
              if (tau != 0.0f) {
                val -= tau * v * (float)dots[col];
              }
              x_next[elem] = val;
            }
          }

          if (elem < total_y_next) {
            int row_next = elem / YCOLS;
            int col = elem - row_next * YCOLS;
            if (n == 1) {
              y_next[elem] = 0.0f;
            } else {
              int row = row_next + 1;
              float v = (float)x[row * p_cur];
              float val = (float)y[row * YCOLS + col];
              if (tau != 0.0f) {
                val -= tau * v * (float)dots[p_cur + col];
              }
              y_next[elem] = val;
            }
          }
        "
      )
    }

    a_work <- x
    y_work <- y
    r_final <- mlx_zeros(c(p, p), dtype = "float32")
    qty_final <- mlx_zeros(c(p, y_cols), dtype = "float32")
    reduce_rows <- block_rows
    threadgroup_threads <- 32L

    for (k in seq_len(p)) {
      current_shape <- mlx_shape(a_work)
      n_current <- as.integer(current_shape[[1L]])
      p_current <- as.integer(current_shape[[2L]])
      reduce_groups <- as.integer(ceiling(n_current / reduce_rows))
      norm_partials <- .mlx_qr_gpu_cache$hh_norm_kernel(
        inputs = list(a_work),
        output_shapes = list(c(reduce_groups)),
        output_dtypes = "float32",
        grid = c(reduce_groups * threadgroup_threads, 1L, 1L),
        threadgroup = c(threadgroup_threads, 1L, 1L),
        template = list(
          K = 0L,
          P = p_current,
          REDUCE_ROWS = reduce_rows,
          THREADS = threadgroup_threads
        )
      )

      params <- .mlx_qr_gpu_cache$hh_params_kernel(
        inputs = list(a_work, norm_partials),
        output_shapes = list(c(3L)),
        output_dtypes = "float32",
        grid = c(threadgroup_threads, 1L, 1L),
        threadgroup = c(threadgroup_threads, 1L, 1L),
        template = list(THREADS = threadgroup_threads)
      )

      target_x_cols <- p_current
      target_cols <- target_x_cols + y_cols
      dot_partials <- .mlx_qr_gpu_cache$hh_dot_kernel(
        inputs = list(a_work, y_work, params),
        output_shapes = list(c(reduce_groups, target_cols)),
        output_dtypes = "float32",
        grid = c(reduce_groups * target_cols * threadgroup_threads, 1L, 1L),
        threadgroup = c(threadgroup_threads, 1L, 1L),
        template = list(
          K = 0L,
          P = p_current,
          YCOLS = y_cols,
          TARGET_X_COLS = target_x_cols,
          TARGET_COLS = target_cols,
          REDUCE_ROWS = reduce_rows,
          THREADS = threadgroup_threads
        )
      )
      dots <- mlx_sum(dot_partials, axes = 1L)

      n_next <- max(n_current - 1L, 1L)
      p_next <- max(p_current - 1L, 1L)
      updated <- .mlx_qr_gpu_cache$hh_compact_update_kernel(
        inputs = list(a_work, y_work, params, dots),
        output_shapes = list(
          c(n_next, p_next),
          c(p_current),
          c(n_next, y_cols),
          c(y_cols)
        ),
        output_dtypes = c("float32", "float32", "float32", "float32"),
        grid = c(max(n_next * p_next, p_current, n_next * y_cols, y_cols),
                 1L, 1L),
        threadgroup = c(256L, 1L, 1L),
        template = list(
          YCOLS = y_cols
        )
      )
      mlx_eval(updated$x_next)
      mlx_eval(updated$r_row)
      mlx_eval(updated$y_next)
      mlx_eval(updated$qty_row)

      r_final <- mlx_slice_update(
        r_final, mlx_reshape(updated$r_row, c(1L, p_current)),
        start = c(k, k), stop = c(k, p)
      )
      qty_final <- mlx_slice_update(
        qty_final, mlx_reshape(updated$qty_row, c(1L, y_cols)),
        start = c(k, 1L), stop = c(k, y_cols)
      )

      if (k < p) {
        a_work <- updated$x_next
        y_work <- updated$y_next
      }
    }

    diag_vals <- as.numeric(diag(r_final))
    row_signs <- sign(diag_vals)
    row_signs[row_signs == 0] <- 1
    for (row in which(row_signs < 0)) {
      r_row <- -r_final[row, , drop = FALSE]
      r_final <- mlx_slice_update(
        r_final, r_row,
        start = c(row, 1L), stop = c(row, p)
      )
      if (has_y) {
        qty_row <- -qty_final[row, , drop = FALSE]
        qty_final <- mlx_slice_update(
          qty_final, qty_row,
          start = c(row, 1L), stop = c(row, y_cols)
        )
      }
    }

    diag_vals <- abs(diag_vals)
    rank_tol <- tol * max(1, max(diag_vals, na.rm = TRUE))
    rank <- sum(is.finite(diag_vals) & diag_vals > rank_tol)
    if (rank < p) {
      stop(sprintf(
        "mlx_qr_gpu() detected rank deficiency: rank %d < %d.",
        rank, p
      ), call. = FALSE)
    }

    out <- list(
      R = r_final,
      rank = rank,
      pivot = seq_len(p),
      block_rows = block_rows,
      method = method
    )
    if (has_y) {
      out$qty <- qty_final
    }
    return(structure(out, class = c("mlx_qr_gpu", "list")))
  }

  if (identical(method, "blocked_householder")) {
    if (is.null(.mlx_qr_gpu_cache$blocked_panel_factor)) {
      .mlx_qr_gpu_cache$blocked_panel_factor <- mlx_compile(function(panel_work) {
        panel_shape <- mlx_shape(panel_work)
        n_panel_rows <- as.integer(panel_shape[[1L]])
        panel_size <- as.integer(panel_shape[[2L]])
        v_mat <- mlx_zeros(c(n_panel_rows, panel_size), dtype = "float32")
        t_mat <- mlx_zeros(c(panel_size, panel_size), dtype = "float32")

        for (j in seq_len(panel_size)) {
          sub_rows <- j:n_panel_rows
          sub_cols <- j:panel_size
          x_tail <- panel_work[sub_rows, j, drop = FALSE]
          norm_x <- sqrt(mlx_sum(x_tail * x_tail))
          alpha <- x_tail[1L, 1L, drop = FALSE]
          beta <- mlx_where(alpha >= 0, -norm_x, norm_x)
          v_tail <- mlx_slice_update(
            x_tail, alpha - beta,
            start = c(1L, 1L), stop = c(1L, 1L)
          )
          tau <- 2 / mlx_sum(v_tail * v_tail)
          v_mat <- mlx_slice_update(
            v_mat, v_tail,
            start = c(j, j), stop = c(n_panel_rows, j)
          )

          panel_tail <- panel_work[sub_rows, sub_cols, drop = FALSE]
          panel_projection <- tau * crossprod(v_tail, panel_tail)
          panel_tail_new <- panel_tail - v_tail %*% panel_projection
          panel_work <- mlx_slice_update(
            panel_work, panel_tail_new,
            start = c(j, j), stop = c(n_panel_rows, panel_size)
          )

          if (j > 1L) {
            prev <- seq_len(j - 1L)
            v_prev <- v_mat[, prev, drop = FALSE]
            t_prev <- t_mat[prev, prev, drop = FALSE]
            v_full <- v_mat[, j, drop = FALSE]
            t_row <- -tau * (crossprod(v_full, v_prev) %*% t_prev)
            t_mat <- mlx_slice_update(
              t_mat, t_row,
              start = c(j, 1L), stop = c(j, j - 1L)
            )
          }
          t_mat <- mlx_slice_update(
            t_mat, tau,
            start = c(j, j), stop = c(j, j)
          )
        }
        list(panel_work = panel_work, v_mat = v_mat, t_mat = t_mat)
      })
    }

    a_work <- x
    y_work <- y
    panel_width <- 16L

    for (panel_start in seq(1L, p, by = panel_width)) {
      panel_end <- min(p, panel_start + panel_width - 1L)
      panel_size <- panel_end - panel_start + 1L
      rows <- panel_start:n
      panel_cols <- panel_start:panel_end
      n_panel_rows <- n - panel_start + 1L

      panel_factor <- .mlx_qr_gpu_cache$blocked_panel_factor(
        a_work[rows, panel_cols, drop = FALSE]
      )
      panel_work <- panel_factor$panel_work
      v_mat <- panel_factor$v_mat
      t_mat <- panel_factor$t_mat

      a_work <- mlx_slice_update(
        a_work, panel_work,
        start = c(panel_start, panel_start), stop = c(n, panel_end)
      )

      if (panel_end < p) {
        after_cols <- (panel_end + 1L):p
        trailing <- a_work[rows, after_cols, drop = FALSE]
        wy_projection <- t_mat %*% crossprod(v_mat, trailing)
        trailing_new <- trailing - v_mat %*% wy_projection
        a_work <- mlx_slice_update(
          a_work, trailing_new,
          start = c(panel_start, panel_end + 1L), stop = c(n, p)
        )
      }

      if (has_y) {
        y_tail <- y_work[rows, , drop = FALSE]
        wy_projection_y <- t_mat %*% crossprod(v_mat, y_tail)
        y_tail_new <- y_tail - v_mat %*% wy_projection_y
        y_work <- mlx_slice_update(
          y_work, y_tail_new,
          start = c(panel_start, 1L), stop = c(n, y_cols)
        )
      }
    }

    r_final <- mlx_triu(a_work[seq_len(p), seq_len(p), drop = FALSE])
    qty_final <- y_work[seq_len(p), , drop = FALSE]

    diag_vals <- as.numeric(diag(r_final))
    row_signs <- sign(diag_vals)
    row_signs[row_signs == 0] <- 1
    for (row in which(row_signs < 0)) {
      r_row <- -r_final[row, , drop = FALSE]
      r_final <- mlx_slice_update(
        r_final, r_row,
        start = c(row, 1L), stop = c(row, p)
      )
      if (has_y) {
        qty_row <- -qty_final[row, , drop = FALSE]
        qty_final <- mlx_slice_update(
          qty_final, qty_row,
          start = c(row, 1L), stop = c(row, y_cols)
        )
      }
    }

    diag_vals <- abs(diag_vals)
    rank_tol <- tol * max(1, max(diag_vals, na.rm = TRUE))
    rank <- sum(is.finite(diag_vals) & diag_vals > rank_tol)
    if (rank < p) {
      stop(sprintf(
        "mlx_qr_gpu() detected rank deficiency: rank %d < %d.",
        rank, p
      ), call. = FALSE)
    }

    out <- list(
      R = r_final,
      rank = rank,
      pivot = seq_len(p),
      block_rows = block_rows,
      method = method
    )
    if (has_y) {
      out$qty <- qty_final
    }
    return(structure(out, class = c("mlx_qr_gpu", "list")))
  }

  if (identical(method, "householder")) {
    a_work <- x
    y_work <- y

    for (k in seq_len(p)) {
      rows <- k:n
      cols <- k:p
      x_tail <- a_work[rows, k, drop = FALSE]
      norm_x <- sqrt(mlx_sum(x_tail * x_tail))
      alpha <- x_tail[1L, 1L, drop = FALSE]
      beta <- mlx_where(alpha >= 0, -norm_x, norm_x)

      # Form the Householder vector u = x - beta e_1. The update
      # H A = A - u (2 / u'u) u' A is applied to the trailing matrix with
      # MLX's optimized reductions and matrix multiplication kernels.
      u <- mlx_slice_update(
        x_tail, alpha - beta,
        start = c(1L, 1L), stop = c(1L, 1L)
      )
      tau <- 2 / mlx_sum(u * u)

      trailing <- a_work[rows, cols, drop = FALSE]
      projection <- tau * crossprod(u, trailing)
      trailing_new <- trailing - u %*% projection
      a_work <- mlx_slice_update(
        a_work, trailing_new,
        start = c(k, k), stop = c(n, p)
      )

      if (has_y) {
        y_tail <- y_work[rows, , drop = FALSE]
        projection_y <- tau * crossprod(u, y_tail)
        y_tail_new <- y_tail - u %*% projection_y
        y_work <- mlx_slice_update(
          y_work, y_tail_new,
          start = c(k, 1L), stop = c(n, y_cols)
        )
      }
    }

    r_final <- mlx_triu(a_work[seq_len(p), seq_len(p), drop = FALSE])
    qty_final <- y_work[seq_len(p), , drop = FALSE]

    diag_vals <- as.numeric(diag(r_final))
    row_signs <- sign(diag_vals)
    row_signs[row_signs == 0] <- 1
    for (row in which(row_signs < 0)) {
      r_row <- -r_final[row, , drop = FALSE]
      r_final <- mlx_slice_update(
        r_final, r_row,
        start = c(row, 1L), stop = c(row, p)
      )
      if (has_y) {
        qty_row <- -qty_final[row, , drop = FALSE]
        qty_final <- mlx_slice_update(
          qty_final, qty_row,
          start = c(row, 1L), stop = c(row, y_cols)
        )
      }
    }

    diag_vals <- abs(diag_vals)
    rank_tol <- tol * max(1, max(diag_vals, na.rm = TRUE))
    rank <- sum(is.finite(diag_vals) & diag_vals > rank_tol)
    if (rank < p) {
      stop(sprintf(
        "mlx_qr_gpu() detected rank deficiency: rank %d < %d.",
        rank, p
      ), call. = FALSE)
    }

    out <- list(
      R = r_final,
      rank = rank,
      pivot = seq_len(p),
      block_rows = block_rows,
      method = method
    )
    if (has_y) {
      out$qty <- qty_final
    }
    return(structure(out, class = c("mlx_qr_gpu", "list")))
  }

  if (is.null(.mlx_qr_gpu_cache$local_kernel)) {
    .mlx_qr_gpu_cache$local_kernel <- mlx_metal_kernel(
      name = "rmlx_qr_gpu_local",
      input_names = c("x", "y"),
      output_names = c("r_out", "qty_out"),
      ensure_row_contiguous = TRUE,
      source = "
        // One Metal threadgroup factors one input tile. Householder reductions
        // are parallel across rows and trailing columns, avoiding a serialized
        // Givens chase for every input row.
        uint block = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint lane = lid & 31;
        uint simdgroup = lid >> 5;
        int n = x_shape[0];
        int row_start = block * BLOCK_ROWS;
        int valid_rows = metal::min(BLOCK_ROWS, n - row_start);

        threadgroup float tile[BLOCK_ROWS * P];
        threadgroup float tile_y[BLOCK_ROWS * YCOLS];
        threadgroup float partial[SIMDGROUPS];
        threadgroup float dots[P + YCOLS];
        threadgroup float params[3];

        for (int i = lid; i < valid_rows * P; i += THREADS) {
          int row = i / P;
          int col = i - row * P;
          tile[i] = (float)x[(row_start + row) * P + col];
        }
        for (int i = lid; i < valid_rows * YCOLS; i += THREADS) {
          int row = i / YCOLS;
          int col = i - row * YCOLS;
          tile_y[i] = (float)y[(row_start + row) * YCOLS + col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int reflectors = metal::min(valid_rows, P);
        for (int k = 0; k < reflectors; ++k) {
          float local_norm = 0.0f;
          for (int row = k + lid; row < valid_rows; row += THREADS) {
            float value = tile[row * P + k];
            local_norm += value * value;
          }
          local_norm = simd_sum(local_norm);
          if (lane == 0) {
            partial[simdgroup] = local_norm;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (simdgroup == 0) {
            float norm_sq = lane < SIMDGROUPS ? partial[lane] : 0.0f;
            norm_sq = simd_sum(norm_sq);
            if (lane == 0) {
              float alpha = tile[k * P + k];
              float norm_x = metal::sqrt(norm_sq);
              float beta = alpha >= 0.0f ? -norm_x : norm_x;
              float v0 = alpha - beta;
              float tail_sq = metal::max(norm_sq - alpha * alpha, 0.0f);
              float v_norm_sq = v0 * v0 + tail_sq;
              params[0] = v0;
              params[1] = v_norm_sq == 0.0f ? 0.0f : 2.0f / v_norm_sq;
              params[2] = beta;
              tile[k * P + k] = v0;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          int target_count = P - k - 1 + YCOLS;
          for (int target = simdgroup; target < target_count;
               target += SIMDGROUPS) {
            float local_dot = 0.0f;
            for (int row = k + lane; row < valid_rows; row += 32) {
              float v = row == k ? params[0] : tile[row * P + k];
              float value;
              if (target < P - k - 1) {
                value = tile[row * P + k + 1 + target];
              } else {
                int y_col = target - (P - k - 1);
                value = tile_y[row * YCOLS + y_col];
              }
              local_dot += v * value;
            }
            local_dot = simd_sum(local_dot);
            if (lane == 0) {
              dots[target] = local_dot;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          int update_size = (valid_rows - k) * target_count;
          for (int i = lid; i < update_size; i += THREADS) {
            int row = k + i / target_count;
            int target = i - (row - k) * target_count;
            float v = row == k ? params[0] : tile[row * P + k];
            float adjustment = params[1] * v * dots[target];
            if (target < P - k - 1) {
              int col = k + 1 + target;
              tile[row * P + col] -= adjustment;
            } else {
              int y_col = target - (P - k - 1);
              tile_y[row * YCOLS + y_col] -= adjustment;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (lid == 0) {
            tile[k * P + k] = params[2];
          }
          for (int row = k + 1 + lid; row < valid_rows; row += THREADS) {
            tile[row * P + k] = 0.0f;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint r_base = block * P * P;
        for (int idx = lid; idx < P * P; idx += THREADS) {
          int row = idx / P;
          int col = idx - row * P;
          if (row <= col && row < valid_rows) {
            float sign = tile[row * P + row] < 0.0f ? -1.0f : 1.0f;
            r_out[r_base + idx] = sign * tile[row * P + col];
          } else {
            r_out[r_base + idx] = 0.0f;
          }
        }

        uint qty_base = block * P * YCOLS;
        for (int idx = lid; idx < P * YCOLS; idx += THREADS) {
          int row = idx / YCOLS;
          int col = idx - row * YCOLS;
          if (row < valid_rows) {
            float sign = tile[row * P + row] < 0.0f ? -1.0f : 1.0f;
            qty_out[qty_base + idx] = sign * tile_y[row * YCOLS + col];
          } else {
            qty_out[qty_base + idx] = 0.0f;
          }
        }
      "
    )
  }

  if (is.null(.mlx_qr_gpu_cache$combine_kernel)) {
    .mlx_qr_gpu_cache$combine_kernel <- mlx_metal_kernel(
      name = "rmlx_qr_gpu_combine",
      input_names = c("r_in", "qty_in"),
      output_names = c("r_out", "qty_out"),
      ensure_row_contiguous = TRUE,
      source = "
        // Combine several compact block QR states into one smaller QR state.
        // One threadgroup owns one output group. It QR-reduces a stack of input
        // R blocks and applies the same rotations to the stacked Q'y vectors.
        uint out_block = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        int in_blocks = r_in_shape[0];
        int first_block = out_block * BLOCKS_PER_GROUP;

        // Shared accumulator for the output group's R and Q'y.
        threadgroup float r[P * P];
        threadgroup float qty[P * YCOLS];
        threadgroup float work[P];
        threadgroup float work_y[YCOLS];
        threadgroup float givens[3];

        // Copy the first triangular factor directly. Only subsequent factors
        // need Givens insertion.
        for (int i = lid; i < P * P; i += THREADS) {
          r[i] = (float)r_in[first_block * P * P + i];
        }
        for (int i = lid; i < P * YCOLS; i += THREADS) {
          qty[i] = (float)qty_in[first_block * P * YCOLS + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int block_offset = 1; block_offset < BLOCKS_PER_GROUP;
             ++block_offset) {
          int in_block = first_block + block_offset;
          if (in_block >= in_blocks) {
            continue;
          }

          for (int in_row = 0; in_row < P; ++in_row) {
            // The input row is triangular, so rotations below in_row are
            // identities and can be skipped.
            uint r_in_base = in_block * P * P + in_row * P;
            for (int col = in_row + lid; col < P; col += THREADS) {
              work[col] = (float)r_in[r_in_base + col];
            }

            uint qty_in_base = in_block * P * YCOLS + in_row * YCOLS;
            for (int col = lid; col < YCOLS; col += THREADS) {
              work_y[col] = (float)qty_in[qty_in_base + col];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int k = in_row; k < P; ++k) {
              // Same Givens insertion as the local kernel: eliminate work[k]
              // against the accumulated diagonal r[k, k].
              if (lid == 0) {
                float a = r[k * P + k];
                float b = work[k];
                float rho = metal::sqrt(a * a + b * b);
                if (rho == 0.0f) {
                  givens[0] = 1.0f;
                  givens[1] = 0.0f;
                  givens[2] = 0.0f;
                } else {
                  givens[0] = a / rho;
                  givens[1] = b / rho;
                  givens[2] = rho;
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);

              // Rotate the upper-triangular accumulator and the pending row.
              float c = givens[0];
              float s = givens[1];
              for (int col = k + 1 + lid; col < P; col += THREADS) {
                float r_old = r[k * P + col];
                float w_old = work[col];
                r[k * P + col] = c * r_old + s * w_old;
                work[col] = -s * r_old + c * w_old;
              }

              // Rotate the accumulated right-hand side in the same way.
              for (int col = lid; col < YCOLS; col += THREADS) {
                float q_old = qty[k * YCOLS + col];
                float y_old = work_y[col];
                qty[k * YCOLS + col] = c * q_old + s * y_old;
                work_y[col] = -s * q_old + c * y_old;
              }

              if (lid == 0) {
                r[k * P + k] = givens[2];
                work[k] = 0.0f;
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
          }
        }

        // Store the reduced group. The R wrapper recursively launches this
        // kernel until only one group remains.
        uint r_base = out_block * P * P;
        for (int idx = lid; idx < P * P; idx += THREADS) {
          int row = idx / P;
          int col = idx - row * P;
          if (row <= col) {
            r_out[r_base + row * P + col] = r[row * P + col];
          } else {
            r_out[r_base + row * P + col] = 0.0f;
          }
        }

        uint qty_base = out_block * P * YCOLS;
        for (int idx = lid; idx < P * YCOLS; idx += THREADS) {
          qty_out[qty_base + idx] = qty[idx];
        }
      "
    )
  }

  n_blocks <- as.integer(ceiling(n / block_rows))
  local_threads <- 256L
  qr_blocks <- .mlx_qr_gpu_cache$local_kernel(
    inputs = list(x, y),
    output_shapes = list(c(n_blocks, p, p), c(n_blocks, p, y_cols)),
    output_dtypes = c("float32", "float32"),
    grid = c(n_blocks * local_threads, 1L, 1L),
    threadgroup = c(local_threads, 1L, 1L),
    template = list(
      P = p,
      YCOLS = y_cols,
      BLOCK_ROWS = block_rows,
      THREADS = local_threads,
      SIMDGROUPS = local_threads %/% 32L
    )
  )

  r_blocks <- qr_blocks$r_out
  qty_blocks <- qr_blocks$qty_out
  current_blocks <- n_blocks
  blocks_per_group <- max(2L, min(8L, as.integer(ceiling(block_rows / p))))
  combine_threads <- 64L

  while (current_blocks > 1L) {
    next_blocks <- as.integer(ceiling(current_blocks / blocks_per_group))
    qr_blocks <- .mlx_qr_gpu_cache$combine_kernel(
      inputs = list(r_blocks, qty_blocks),
      output_shapes = list(c(next_blocks, p, p), c(next_blocks, p, y_cols)),
      output_dtypes = c("float32", "float32"),
      grid = c(next_blocks * combine_threads, 1L, 1L),
      threadgroup = c(combine_threads, 1L, 1L),
      template = list(
        P = p,
        YCOLS = y_cols,
        BLOCKS_PER_GROUP = blocks_per_group,
        THREADS = combine_threads
      )
    )
    r_blocks <- qr_blocks$r_out
    qty_blocks <- qr_blocks$qty_out
    current_blocks <- next_blocks
  }

  r_final <- mlx_reshape(r_blocks, c(p, p))
  qty_final <- mlx_reshape(qty_blocks, c(p, y_cols))
  diag_vals <- abs(as.numeric(diag(r_final)))
  rank_tol <- tol * max(1, max(diag_vals, na.rm = TRUE))
  rank <- sum(is.finite(diag_vals) & diag_vals > rank_tol)
  if (rank < p) {
    stop(sprintf(
      "mlx_qr_gpu() detected rank deficiency: rank %d < %d.",
      rank, p
    ), call. = FALSE)
  }

  out <- list(
    R = r_final,
    rank = rank,
    pivot = seq_len(p),
    block_rows = block_rows,
    method = method
  )
  if (has_y) {
    out$qty <- qty_final
  }
  structure(out, class = c("mlx_qr_gpu", "list"))
}

#' Singular value decomposition
#'
#' Generic function for SVD computation.
#' @param x An object.
#' @inheritParams ellipsis_base
#' @return A list with components `d`, `u`, and `v`.
#' @export
svd <- function(x, ...) {
  UseMethod("svd")
}

#' @export
svd.default <- function(x, ...) base::svd(x, ...)

#' Singular value decomposition for mlx arrays
#'
#' Note that mlx's svd returns "full" SVD, with U and V' both square matrices.
#' This is different from R's implementation.
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_matrix_required
#' @param nu Number of left singular vectors to return (0 or `min(dim(x))`).
#' @param nv Number of right singular vectors to return (0 or `min(dim(x))`).
#' @inheritParams ellipsis_ignored
#' @inheritParams common_params
#' @return A list with components `d`, `u`, and `v`.
#' @seealso [mlx.linalg.svd](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.svd)
#' @export
#' @examples
#' x <- mlx_matrix(c(1, 0, 0, 2), 2, 2)
#' svd(x, device = "cpu")
svd.mlx <- function(x, nu = min(n, p), nv = min(n, p), ..., device = NULL) {
  x <- as_mlx(x)
  stopifnot(length(dim(x)) == 2L)
  if (!is.null(device)) local_device(device)

  n <- dim(x)[1]
  p <- dim(x)[2]
  full_cols <- min(n, p)

  if (!nu %in% c(0L, full_cols)) {
    stop("svd.mlx only supports nu = 0 or nu = min(nrow, ncol).", call. = FALSE)
  }
  if (!nv %in% c(0L, full_cols)) {
    stop("svd.mlx only supports nv = 0 or nv = min(nrow, ncol).", call. = FALSE)
  }

  compute_uv <- (nu > 0 || nv > 0)
  x_dtype <- mlx_dtype(x)
  res <- cpp_mlx_svd(x$ptr, compute_uv, x_dtype)

  if (!compute_uv) {
    s_ptr <- res[[1L]]
    d <- new_mlx(s_ptr)
    return(list(d = d, u = NULL, v = NULL))
  }

  U <- new_mlx(res$U)
  S <- new_mlx(res$S)
  Vh <- new_mlx(res$Vh)

  d <- S
  V <- new_mlx(cpp_mlx_transpose(Vh$ptr))

  list(d = d, u = U, v = V)
}

#' Moore-Penrose pseudoinverse for MLX arrays
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @param x An mlx object or coercible matrix.
#' @inheritParams common_params
#' @return An mlx object containing the pseudoinverse.
#' @seealso [mlx.linalg.pinv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.pinv)
#' @export
#' @examples
#' x <- mlx_matrix(c(1, 2, 3, 4), 2, 2)
#' pinv(x, device = "cpu")
pinv <- function(x, device = NULL) {
  x <- as_mlx(x)
  if (!is.null(device)) local_device(device)
  stopifnot(length(dim(x)) == 2L)

  x_dtype <- mlx_dtype(x)
  ptr <- cpp_mlx_pinv(x$ptr, x_dtype)
  new_mlx(ptr, dimnames = dimnames_matrix_inverse(x))
}

#' Fast Fourier Transform
#'
#' Extends [stats::fft()] to work with mlx objects while delegating to the
#' standard R implementation for other inputs.
#'
#' @param z Input to transform. May be a numeric, complex, or mlx object.
#' @param inverse Logical flag; if `TRUE` compute the inverse transform.
#' @inheritParams common_params
#' @inheritParams ellipsis_base
#' @return For mlx inputs, an mlx object containing complex frequency
#'   coefficients; otherwise the base R result.
#' @seealso [stats::fft()], [mlx_fft()], [mlx_fft2()], [mlx_fftn()], [mlx.core.fft.fft](https://ml-explore.github.io/mlx/build/html/python/fft.html#mlx.core.fft.fft)
#' @export
#' @examples
#' z <- as_mlx(c(1, 2, 3, 4))
#' fft(z)
#' fft(z, inverse = TRUE)
fft <- function(z, inverse = FALSE, ...) {
  UseMethod("fft")
}

#' @export
#' @rdname fft
fft.default <- function(z, inverse = FALSE, ...) {
  stats::fft(z, inverse = inverse, ...)
}

#' @export
#' @rdname fft
fft.mlx <- function(z, inverse = FALSE, axis, ...) {
  mlx_fft(z, axis = axis, inverse = inverse)
}

#' Matrix and vector norms for mlx arrays
#'
#' @inheritParams mlx_array_required
#' @param ord Numeric or character norm order. Use `NULL` for the default 2-norm.
#' @inheritParams common_params
#' @return An mlx array containing the requested norm.
#' @seealso [mlx.linalg.norm](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.norm)
#' @export
#' @examples
#' x <- mlx_matrix(1:4, 2, 2)
#' mlx_norm(x)
#' mlx_norm(x, ord = 2)
#' mlx_norm(x, axes = 2)
mlx_norm <- function(x, ord = NULL, axes = NULL, drop = TRUE) {
  x <- as_mlx(x)

  if (!is.null(ord) && !is.numeric(ord) && !is.character(ord)) {
    stop("ord must be numeric, character, or NULL.", call. = FALSE)
  }
  if (is.character(ord) && length(ord) == 1L) {
    ord <- toupper(ord)
  }
  axes_arg <- if (is.null(axes)) NULL else as.integer(axes)
  ptr <- cpp_mlx_norm(x$ptr, ord, axes_arg, !isTRUE(drop))
  new_mlx(ptr, dimnames = dimnames_reduction(x, axes_arg, drop))
}

#' Eigen decomposition for mlx arrays
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_matrix_required
#' @inheritParams common_params
#' @return A list with components `values` and `vectors`, both mlx arrays.
#' @seealso [mlx.linalg.eig](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eig)
#' @export
#' @examples
#' x <- mlx_matrix(c(2, -1, 0, 2), 2, 2)
#' eig <- mlx_eig(x, device = "cpu")
#' eig$values
#' eig$vectors
mlx_eig <- function(x, device = NULL) {
  x <- as_mlx(x)
  if (!is.null(device)) local_device(device)
  stopifnot(length(dim(x)) == 2L, dim(x)[1] == dim(x)[2])

  res <- cpp_mlx_eig(x$ptr)
  list(
    values = new_mlx(res$values),
    vectors = new_mlx(res$vectors)
  )
}

#' Eigenvalues of mlx arrays
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_eig
#' @inheritParams common_params
#' @return An mlx array containing eigenvalues.
#' @seealso [mlx.linalg.eigvals](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigvals)
#' @export
#' @examples
#' x <- mlx_matrix(c(3, 1, 0, 2), 2, 2)
#' mlx_eigvals(x, device = "cpu")
mlx_eigvals <- function(x, device = NULL) {
  x <- as_mlx(x)
  if (!is.null(device)) local_device(device)
  stopifnot(length(dim(x)) == 2L, dim(x)[1] == dim(x)[2])
  ptr <- cpp_mlx_eigvals(x$ptr)
  new_mlx(ptr)
}

#' Eigenvalues of Hermitian mlx arrays
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_eig
#' @inheritParams common_params
#' @param uplo Character string indicating which triangle to use ("L" or "U").
#' @return An mlx array containing eigenvalues.
#' @seealso [mlx.linalg.eigvalsh](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigvalsh)
#' @export
#' @examples
#' x <- mlx_matrix(c(2, 1, 1, 3), 2, 2)
#' mlx_eigvalsh(x, device = "cpu")
mlx_eigvalsh <- function(x, uplo = c("L", "U"), device = NULL) {
  x <- as_mlx(x)
  if (!is.null(device)) local_device(device)
  stopifnot(length(dim(x)) == 2L, dim(x)[1] == dim(x)[2])
  uplo <- match.arg(uplo)
  ptr <- cpp_mlx_eigvalsh(x$ptr, uplo)
  new_mlx(ptr)
}

#' Eigen decomposition of Hermitian mlx arrays
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_eigvalsh
#' @inheritParams common_params
#' @return A list with components `values` and `vectors`.
#' @seealso [mlx.linalg.eigh](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigh)
#' @export
#' @examples
#' x <- mlx_matrix(c(2, 1, 1, 3), 2, 2)
#' mlx_eigh(x, device = "cpu")
mlx_eigh <- function(x, uplo = c("L", "U"), device = NULL) {
  x <- as_mlx(x)
  if (!is.null(device)) local_device(device)
  stopifnot(length(dim(x)) == 2L, dim(x)[1] == dim(x)[2])
  uplo <- match.arg(uplo)
  res <- cpp_mlx_eigh(x$ptr, uplo)
  list(
    values = new_mlx(res$values),
    vectors = new_mlx(res$vectors)
  )
}

#' Solve triangular systems with mlx arrays
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @param a An mlx triangular matrix.
#' @param b Right-hand side matrix or vector.
#' @param upper Logical; if `TRUE`, `a` is upper triangular, otherwise lower.
#' @inheritParams ellipsis_base
#' @inheritParams common_params
#' @return An mlx array solution.
#' @seealso [mlx.linalg.solve_triangular](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.solve_triangular)
#' @export
#' @examples
#' a <- mlx_matrix(c(2, 1, 0, 3), 2, 2)
#' b <- mlx_matrix(c(1, 5), 2, 1)
#' mlx_solve_triangular(a, b, upper = FALSE, device = "cpu")
mlx_solve_triangular <- function(a, b, upper = FALSE, device = NULL) {
  if (!is.null(device)) local_device(device)
  a <- as_mlx(a)
  b <- as_mlx(b)
  stopifnot(length(dim(a)) == 2L, dim(a)[1] == dim(a)[2])
  ptr <- cpp_mlx_solve_triangular(a$ptr, b$ptr, upper)
  new_mlx(ptr, dimnames = dimnames_solve(a, b))
}

#' @rdname mlx_solve_triangular
#' @param r Triangular system matrix passed to [backsolve()].
#' @param x Right-hand side supplied to [backsolve()].
#' @param k Number of columns of `r` to use.
#' @param upper.tri Logical; indicates if `r` is upper triangular.
#' @param transpose Logical; if `TRUE`, solve `t(r) %*% x = b`.
#' @export
backsolve <- function(r, x, k = NULL, upper.tri = TRUE, transpose = FALSE, ...) {
  UseMethod("backsolve")
}

#' @rdname mlx_solve_triangular
#' @export
backsolve.default <- function(r, x, k = NULL, upper.tri = TRUE, transpose = FALSE, ...) {
  base::backsolve(r, x, k = if (is.null(k)) ncol(r) else k, upper.tri = upper.tri, transpose = transpose, ...)
}

#' @rdname mlx_solve_triangular
#' @export
backsolve.mlx <- function(r, x, k = NULL, upper.tri = TRUE, transpose = FALSE, ...,
                          device = NULL) {
  r_mlx <- as_mlx(r)
  x_mlx <- as_mlx(x)

  if (length(dim(r_mlx)) != 2L) {
    stop("`r` must be a matrix when using backsolve() with mlx arrays.", call. = FALSE)
  }

  if (is.null(k)) {
    k <- dim(r_mlx)[2L]
  }
  if (!identical(k, dim(r_mlx)[2L])) {
    stop("`k` values other than ncol(r) are not yet supported for mlx arrays.", call. = FALSE)
  }

  target <- r_mlx
  if (transpose) {
    target <- t(target)
    upper.tri <- !upper.tri
  }

  out <- mlx_solve_triangular(target, x_mlx, upper = upper.tri, device = device)
  dimnames(out) <- NULL
  out
}

#' Vector cross product with mlx arrays
#'
#' @param a,b Input mlx arrays containing 3D vectors.
#' @param axis Axis along which to compute the cross product (1-indexed).
#'   Omit the argument to use the trailing dimension.
#' @return An mlx array of cross products.
#' @seealso [mlx.linalg.cross](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.cross)
#' @export
#' @examples
#' u <- as_mlx(c(1, 0, 0))
#' v <- as_mlx(c(0, 1, 0))
#' mlx_cross(u, v)
mlx_cross <- function(a, b, axis = NULL) {
  a <- as_mlx(a)
  b <- as_mlx(b)
  shape_len <- length(mlx_shape(a))
  axis_val <- if (missing(axis) || is.null(axis)) {
    if (shape_len == 0L) {
      stop("`axis` must be supplied for scalar inputs.", call. = FALSE)
    }
    shape_len
  } else axis
  if (length(axis_val) != 1L || is.na(axis_val) || axis_val < 1L) {
    stop("`axis` must be NULL or a single positive integer.", call. = FALSE)
  }
  ptr <- cpp_mlx_cross(a$ptr, b$ptr, as.integer(axis_val))
  new_mlx(ptr)
}

#' Matrix trace for mlx arrays
#'
#' Computes the sum of the diagonal elements of a 2D array, or the sum along
#' diagonals of a higher dimensional array.
#'
#' @inheritParams mlx_array_required
#' @param offset Offset of the diagonal (0 for main diagonal, positive for above, negative for below).
#' @param axis1,axis2 Axes along which the diagonals are taken (1-indexed, default 1 and 2).
#' @return An mlx scalar or array containing the trace.
#' @seealso [mlx.core.trace](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.trace.html)
#' @export
#' @examples
#' x <- mlx_matrix(1:9, 3, 3)
#' mlx_trace(x)
#' mlx_trace(x, offset = 1)
mlx_trace <- function(x, offset = 0L, axis1 = 1L, axis2 = 2L) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_trace(x$ptr, as.integer(offset), as.integer(axis1), as.integer(axis2))
  new_mlx(ptr)
}

#' Extract diagonal or construct diagonal matrix for mlx arrays
#'
#' Extract a diagonal from a matrix or construct a diagonal matrix from a vector.
#'
#' @param x An mlx array. If 1D, creates a diagonal matrix. If 2D or higher, extracts the diagonal.
#' @param offset Diagonal offset (0 for main diagonal, positive for above, negative for below).
#' @param axis1,axis2 For multi-dimensional arrays, which axes define the 2D planes (1-indexed).
#' @return An mlx array.
#' @seealso [mlx.core.diagonal](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.diagonal.html)
#' @export
#' @examples
#' # Extract diagonal
#' x <- mlx_matrix(1:9, 3, 3)
#' mlx_diagonal(x)
#' # (Constructing diagonals from 1D inputs is not yet supported.)
mlx_diagonal <- function(x, offset = 0L, axis1 = 1L, axis2 = 2L) {
  x <- as_mlx(x)
  offset <- as.integer(offset)
  axis1 <- normalize_axis_single(as.integer(axis1), x) + 1L
  axis2 <- normalize_axis_single(as.integer(axis2), x) + 1L
  ptr <- cpp_mlx_diagonal(x$ptr, offset, axis1, axis2)
  out <- new_mlx(ptr)
  dimnames(out) <- dimnames_diagonal(x, mlx_shape(out), offset, axis1, axis2)
  out
}

#' Outer product of two vectors
#'
#' @param X,Y Numeric vectors or mlx arrays.
#' @param FUN Function to apply (for default method).
#' @inheritParams ellipsis_base
#' @return For mlx inputs, an mlx matrix. Otherwise delegates to `base::outer`.
#' @seealso [mlx.core.outer](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.outer.html)
#' @export
#' @examples
#' x <- as_mlx(c(1, 2, 3))
#' y <- as_mlx(c(4, 5))
#' outer(x, y)
outer <- function(X, Y, FUN = "*", ...) {
  UseMethod("outer")
}

#' @export
outer.default <- base::outer

#' @export
#' @rdname outer
outer.mlx <- function(X, Y, FUN = "*", ...) {
  X <- as_mlx(X)
  Y <- as_mlx(Y)
  ptr <- cpp_mlx_outer(X$ptr, Y$ptr)
  new_mlx(ptr, dimnames = list(names(X), names(Y)))
}

#' Unflatten an axis into multiple axes
#'
#' The reverse of flattening: expands a single axis into multiple axes with the given shape.
#'
#' @inheritParams mlx_array_required
#' @param axis Which axis to unflatten (1-indexed).
#' @param shape Integer vector specifying the new shape for the unflattened axis.
#' @return An mlx array with the axis expanded.
#' @seealso [mlx.core.unflatten](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.unflatten.html)
#' @export
#' @examples
#' # Flatten and unflatten
#' x <- mlx_array(1:24, c(2, 3, 4))
#' x_flat <- mlx_reshape(x, c(2, 12))  # flatten last two dims
#' mlx_unflatten(x_flat, axis = 2, shape = c(3, 4))  # restore original shape
mlx_unflatten <- function(x, axis, shape) {
  x <- as_mlx(x)
  ptr <- cpp_mlx_unflatten(x$ptr, as.integer(axis), as.integer(shape))
  new_mlx(ptr)
}

#' Compute matrix inverse
#'
#' Computes the inverse of a square matrix. Note that as of MLX 0.30.0, this
#' runs on the CPU.
#'
#' @inheritParams mlx_array_required
#' @inherit mlx_cpu_only_operation details
#' @inheritParams common_params
#' @return The inverse of `x`.
#' @seealso [mlx.core.linalg.inv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.inv)
#' @export
#' @examples
#' A <- mlx_matrix(c(4, 7, 2, 6), 2, 2)
#' A_inv <- mlx_inv(A, device = "cpu")
#' # Verify: A %*% A_inv should be identity
#' A %*% A_inv
mlx_inv <- function(x, device = NULL) {
  x <- as_mlx(x)
  if (!is.null(device)) local_device(device)
  ptr <- cpp_mlx_inv(x$ptr)
  new_mlx(ptr, dimnames = dimnames_matrix_inverse(x))
}

#' Compute triangular matrix inverse
#'
#' Computes the inverse of a triangular matrix.
#'
#' **Note:** MLX may crash if `x` is not triangular.
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_array_required
#' @param upper Logical; if `TRUE`, `x` is upper triangular, otherwise lower triangular.
#' @inheritParams common_params
#' @return The inverse of the triangular matrix `x`.
#' @seealso [mlx.core.linalg.tri_inv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.tri_inv)
#' @export
#' @examples
#' # Lower triangular matrix
#' L <- mlx_matrix(c(1:3, 0, 4:5, 0, 0, 6), 3, 3)
#' mlx_tri_inv(L, upper = FALSE, device = "cpu")
mlx_tri_inv <- function(x, upper = FALSE, device = NULL) {
  x <- as_mlx(x)
  if (!is.null(device)) local_device(device)
  ptr <- cpp_mlx_tri_inv(x$ptr, upper)
  new_mlx(ptr, dimnames = dimnames_matrix_inverse(x))
}

#' Compute matrix inverse via Cholesky decomposition
#'
#' Computes the inverse of a positive definite matrix from its Cholesky factor.
#' Note: `x` should be the Cholesky factor (L or U), not the original matrix.
#'
#' For a more R-like interface, see [chol2inv()].
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_array_required
#' @param upper Logical; if `TRUE`, `x` is upper triangular, otherwise lower triangular.
#' @inheritParams common_params
#' @return The inverse of the original matrix (A^-1 where A = LL' or A = U'U).
#' @seealso [chol2inv()], [mlx.core.linalg.cholesky_inv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.cholesky_inv)
#' @export
#' @examples
#' # Create a positive definite matrix
#' A <- matrix(rnorm(9), 3, 3)
#' A <- t(A) %*% A
#' # Compute Cholesky factor
#' L <- chol(A, pivot = FALSE, upper = FALSE)
#' # Get inverse from Cholesky factor
#' mlx_cholesky_inv(as_mlx(L), device = "cpu")
mlx_cholesky_inv <- function(x, upper = FALSE, device = NULL) {
  x <- as_mlx(x)
  if (!is.null(device)) local_device(device)
  ptr <- cpp_mlx_cholesky_inv(x$ptr, upper)
  new_mlx(ptr)
}

#' LU factorization
#'
#' Computes the LU factorization of a matrix.
#'
#' @inherit mlx_cpu_only_operation details
#'
#' @inheritParams mlx_array_required
#' @inheritParams common_params
#' @return A list with components `p` (pivot indices), `l` (lower triangular),
#'   and `u` (upper triangular). The relationship is `A = L[P, ] %*% U`.
#' @seealso [mlx.core.linalg.lu](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.lu)
#' @export
#' @examples
#' A <- mlx_matrix(rnorm(16), 4, 4)
#' lu_result <- mlx_lu(A, device = "cpu")
#' P <- lu_result$p  # Pivot indices
#' L <- lu_result$l  # Lower triangular
#' U <- lu_result$u  # Upper triangular
mlx_lu <- function(x, device = NULL) {
  x <- as_mlx(x)
  if (!is.null(device)) local_device(device)
  result <- cpp_mlx_lu(x$ptr)
  list(
    p = new_mlx(result$p),
    l = new_mlx(result$l),
    u = new_mlx(result$u)
  )
}
