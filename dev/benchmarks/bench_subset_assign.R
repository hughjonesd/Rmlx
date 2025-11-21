# Benchmark subset assignment performance for Rmlx
#
# Scenarios:
# - Large vector with R logical mask vs mlx logical mask
# - Large matrix with full logical mask (R vs mlx)
# - Matrix with row/col logical masks (R vs mlx)
# For each scenario, compare with/without mlx_eval() on the result.

suppresPackageStartupMessages <- suppressPackageStartupMessages
suppresPackageStartupMessages({
  library(devtools)
  library(bench)
  library(rlang)
  library(stats) # for setNames
  library(Rmlx)
})

load_all(quiet = TRUE)

set.seed(20251121)

# Dataset sizes (keep CPU/GPU friendly but representative)
n_vec <- 2e6                # ~16 MB double vector
mat_nrow <- 2000
mat_ncol <- 1000            # ~16 MB matrix

base_vec <- runif(n_vec)
base_mat <- matrix(runif(mat_nrow * mat_ncol), nrow = mat_nrow, ncol = mat_ncol)

# Logical masks
mask_r_vec <- base_vec > 0.5
mask_mlx_vec <- as_mlx(mask_r_vec)

mask_r_mat <- base_mat > 0.5
mask_mlx_mat <- as_mlx(mask_r_mat)

row_mask_r <- sample(c(TRUE, FALSE), size = mat_nrow, replace = TRUE, prob = c(0.5, 0.5))
col_mask_r <- sample(c(TRUE, FALSE), size = mat_ncol, replace = TRUE, prob = c(0.5, 0.5))
row_mask_mlx <- as_mlx(row_mask_r)
col_mask_mlx <- as_mlx(col_mask_r)

# Numeric row indices (non-contiguous) and contiguous slice
rows_noncontig_r <- c(2L, 4L, 5L, 20L, 50L, 100L, 500L, 1500L)
rows_noncontig_mlx <- as_mlx(rows_noncontig_r)
rows_contig_r <- 501:1500
rows_contig_mlx <- as_mlx(rows_contig_r)
rows_dense_noncontig_r <- sample(501:1500)
rows_dense_noncontig_mlx <- as_mlx(rows_dense_noncontig_r)
# Medium-sparse numeric: 1000 random rows across full range (50% density)
rows_medium_r <- sort(sample(mat_nrow, 1000))
rows_medium_mlx <- as_mlx(rows_medium_r)

with_slice_enabled <- function(enabled, expr) {
  ns <- asNamespace("Rmlx")
  old_fun <- get(".mlx_slice_parameters", envir = ns)
  unlockBinding(".mlx_slice_parameters", ns)
  on.exit({
    unlockBinding(".mlx_slice_parameters", ns)
    assign(".mlx_slice_parameters", old_fun, envir = ns)
    lockBinding(".mlx_slice_parameters", ns)
  }, add = TRUE)
  if (enabled) {
    assign(".mlx_slice_parameters", old_fun, envir = ns)
  } else {
    assign(".mlx_slice_parameters", function(...) list(can_slice = FALSE), envir = ns)
  }
  lockBinding(".mlx_slice_parameters", ns)
  force(expr)
}

cases <- list(
  vector_R_mask_no_eval = quote({
    x <- as_mlx(base_vec)
    x[mask_r_vec] <- 0
  }),
  vector_R_mask_eval = quote({
    x <- as_mlx(base_vec)
    x[mask_r_vec] <- 0
    mlx_eval(x)
  }),
  vector_mlx_mask_no_eval = quote({
    x <- as_mlx(base_vec)
    x[mask_mlx_vec] <- 0
  }),
  vector_mlx_mask_eval = quote({
    x <- as_mlx(base_vec)
    x[mask_mlx_vec] <- 0
    mlx_eval(x)
  }),
  matrix_full_R_mask_no_eval = quote({
    x <- as_mlx(base_mat)
    x[mask_r_mat] <- 1
  }),
  matrix_full_R_mask_eval = quote({
    x <- as_mlx(base_mat)
    x[mask_r_mat] <- 1
    mlx_eval(x)
  }),
  matrix_full_mlx_mask_no_eval = quote({
    x <- as_mlx(base_mat)
    x[mask_mlx_mat] <- 1
  }),
  matrix_full_mlx_mask_eval = quote({
    x <- as_mlx(base_mat)
    x[mask_mlx_mat] <- 1
    mlx_eval(x)
  }),
  matrix_rowcol_R_mask_no_eval = quote({
    x <- as_mlx(base_mat)
    x[row_mask_r, col_mask_r] <- 2
  }),
  matrix_rowcol_R_mask_eval = quote({
    x <- as_mlx(base_mat)
    x[row_mask_r, col_mask_r] <- 2
    mlx_eval(x)
  }),
  matrix_rowcol_mlx_mask_no_eval = quote({
    x <- as_mlx(base_mat)
    x[row_mask_mlx, col_mask_mlx] <- 2
  }),
  matrix_rowcol_mlx_mask_eval = quote({
    x <- as_mlx(base_mat)
    x[row_mask_mlx, col_mask_mlx] <- 2
    mlx_eval(x)
  }),
  matrix_rows_numeric_R_no_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_noncontig_r, ] <- 3
  }),
  matrix_rows_numeric_R_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_noncontig_r, ] <- 3
    mlx_eval(x)
  }),
  matrix_rows_numeric_mlx_no_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_noncontig_mlx, ] <- 3
  }),
  matrix_rows_numeric_mlx_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_noncontig_mlx, ] <- 3
    mlx_eval(x)
  }),
  matrix_rows_contig_R_no_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_contig_r, ] <- 4
  }),
  matrix_rows_contig_R_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_contig_r, ] <- 4
    mlx_eval(x)
  }),
  matrix_rows_contig_mlx_no_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_contig_mlx, ] <- 4
  }),
  matrix_rows_contig_mlx_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_contig_mlx, ] <- 4
    mlx_eval(x)
  }),
  matrix_rows_contig_R_scatter = quote({
    with_slice_enabled(FALSE, {
      x <- as_mlx(base_mat)
      x[rows_contig_r, ] <- 4
    })
  }),
  matrix_rows_contig_R_scatter_eval = quote({
    with_slice_enabled(FALSE, {
      x <- as_mlx(base_mat)
      x[rows_contig_r, ] <- 4
      mlx_eval(x)
    })
  }),
  matrix_rows_contig_mlx_scatter = quote({
    with_slice_enabled(FALSE, {
      x <- as_mlx(base_mat)
      x[rows_contig_mlx, ] <- 4
    })
  }),
  matrix_rows_contig_mlx_scatter_eval = quote({
    with_slice_enabled(FALSE, {
      x <- as_mlx(base_mat)
      x[rows_contig_mlx, ] <- 4
      mlx_eval(x)
    })
  }),
  matrix_rows_dense_noncontig_R_no_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_dense_noncontig_r, ] <- 6
  }),
  matrix_rows_dense_noncontig_R_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_dense_noncontig_r, ] <- 6
    mlx_eval(x)
  }),
  matrix_rows_dense_noncontig_mlx_no_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_dense_noncontig_mlx, ] <- 6
  }),
  matrix_rows_dense_noncontig_mlx_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_dense_noncontig_mlx, ] <- 6
    mlx_eval(x)
  }),
  matrix_rows_medium_R_no_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_medium_r, ] <- 7
  }),
  matrix_rows_medium_R_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_medium_r, ] <- 7
    mlx_eval(x)
  }),
  matrix_rows_medium_mlx_no_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_medium_mlx, ] <- 7
  }),
  matrix_rows_medium_mlx_eval = quote({
    x <- as_mlx(base_mat)
    x[rows_medium_mlx, ] <- 7
    mlx_eval(x)
  })
)

exprs <- rlang::exprs(!!!cases)

marks <- do.call(
  bench::mark,
  c(
    exprs,
    list(
      iterations = 7,
      check = FALSE,
      memory = FALSE,
      time_unit = "s"
    )
  )
)

res <- data.frame(
  case = as.character(marks$expression),
  median_sec = marks$median,
  itr_per_sec = marks$`itr/sec`,
  stringsAsFactors = FALSE
)

res$eval <- grepl("_eval$", res$case)

res$domain <- ifelse(grepl("^vector", res$case), "vector",
              ifelse(grepl("^matrix_full", res$case), "matrix-full-mask",
              ifelse(grepl("^matrix_rowcol", res$case), "matrix-rowcol-mask",
              ifelse(grepl("^matrix_rows_dense", res$case), "matrix-rows-dense",
              ifelse(grepl("^matrix_rows_medium", res$case), "matrix-rows-medium",
              ifelse(grepl("^matrix_rows_noncontig", res$case), "matrix-rows-sparse",
              ifelse(grepl("^matrix_rows_contig", res$case), "matrix-rows-contig", "other")))))))

res$index <- ifelse(grepl("mask_mlx", res$case), "mlx logical",
             ifelse(grepl("mask_R", res$case), "R logical",
             ifelse(grepl("numeric_mlx", res$case), "mlx numeric",
             ifelse(grepl("numeric_R", res$case), "R numeric",
             ifelse(grepl("mask", res$case), "logical", "numeric")))))

res$path <- ifelse(grepl("scatter", res$case), "numeric scatter (slice off)",
             ifelse(grepl("mask", res$case), "boolean mask",
             ifelse(grepl("contig", res$case), "numeric slice (fast path)", "numeric")))

res <- res[order(res$domain, res$path, res$eval, res$median_sec), ]
rownames(res) <- NULL

pretty_print <- function(df, title) {
  cat("\n== ", title, " ==\n", sep = "")
  print(df[, c("case", "path", "index", "eval", "median_sec", "itr_per_sec")], row.names = FALSE)
}

split(res, res$domain) |>
  lapply(function(df) pretty_print(df, unique(df$domain))) |>
  invisible()
