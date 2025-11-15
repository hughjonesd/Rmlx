#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(devtools)
})

devtools::load_all(".", quiet = TRUE)

run_case <- function(fn, reps = 3L) {
  times <- numeric(reps)
  for (i in seq_len(reps)) {
    gc()
    times[i] <- system.time(fn())["elapsed"]
  }
  c(mean = mean(times), median = median(times), min = min(times), max = max(times))
}

run_assign_benchmarks <- function(iterations = 3L, seed = 42L) {
  set.seed(seed)

  vec_len <- 4000000L
  vec <- as_mlx(runif(vec_len))
  vec_device <- vec$device
  vec_idx_contig <- seq_len(vec_len * 3L %/% 5L)
  vec_idx_noncontig <- sample(vec_len, length(vec_idx_contig))

  vec_idx_contig_mlx <- as_mlx(vec_idx_contig)
  vec_idx_noncontig_mlx <- as_mlx(vec_idx_noncontig)

  vec_val_contig <- runif(length(vec_idx_contig))
  vec_val_noncontig <- runif(length(vec_idx_noncontig))

  mat_dim <- c(1500L, 1200L)
  mat <- as_mlx(matrix(runif(prod(mat_dim)), nrow = mat_dim[1], ncol = mat_dim[2]))
  mat_device <- mat$device

  row_idx_contig <- seq_len(mat_dim[1] - 200L)
  col_idx_contig <- seq_len(mat_dim[2] - 150L)

  row_idx_noncontig <- sort(sample(mat_dim[1], length(row_idx_contig)))
  col_idx_noncontig <- sort(sample(mat_dim[2], length(col_idx_contig)))

  row_idx_contig_mlx <- as_mlx(row_idx_contig)
  col_idx_contig_mlx <- as_mlx(col_idx_contig)
  row_idx_noncontig_mlx <- as_mlx(row_idx_noncontig)
  col_idx_noncontig_mlx <- as_mlx(col_idx_noncontig)

  mat_val_contig <- matrix(runif(length(row_idx_contig) * length(col_idx_contig)),
                           nrow = length(row_idx_contig),
                           ncol = length(col_idx_contig))
  mat_val_noncontig <- matrix(runif(length(row_idx_noncontig) * length(col_idx_noncontig)),
                              nrow = length(row_idx_noncontig),
                              ncol = length(col_idx_noncontig))

  cases <- list(
    vector_contig_r = function() {
      tmp <- vec
      tmp[vec_idx_contig] <- vec_val_contig
      mlx_synchronize(vec_device)
      invisible(tmp)
    },
    vector_noncontig_r = function() {
      tmp <- vec
      tmp[vec_idx_noncontig] <- vec_val_noncontig
      mlx_synchronize(vec_device)
      invisible(tmp)
    },
    vector_contig_mlx = function() {
      tmp <- vec
      tmp[vec_idx_contig_mlx] <- vec_val_contig
      mlx_synchronize(vec_device)
      invisible(tmp)
    },
    vector_noncontig_mlx = function() {
      tmp <- vec
      tmp[vec_idx_noncontig_mlx] <- vec_val_noncontig
      mlx_synchronize(vec_device)
      invisible(tmp)
    },
    matrix_contig_r = function() {
      tmp <- mat
      tmp[row_idx_contig, col_idx_contig] <- mat_val_contig
      mlx_synchronize(mat_device)
      invisible(tmp)
    },
    matrix_noncontig_r = function() {
      tmp <- mat
      tmp[row_idx_noncontig, col_idx_noncontig] <- mat_val_noncontig
      mlx_synchronize(mat_device)
      invisible(tmp)
    },
    matrix_contig_mlx = function() {
      tmp <- mat
      tmp[row_idx_contig_mlx, col_idx_contig_mlx] <- mat_val_contig
      mlx_synchronize(mat_device)
      invisible(tmp)
    },
    matrix_noncontig_mlx = function() {
      tmp <- mat
      tmp[row_idx_noncontig_mlx, col_idx_noncontig_mlx] <- mat_val_noncontig
      mlx_synchronize(mat_device)
      invisible(tmp)
    }
  )

  results <- lapply(cases, run_case, reps = iterations)
  df <- do.call(rbind, results)
  df <- data.frame(
    scenario = names(results),
    mean = df[, "mean"],
    median = df[, "median"],
    min = df[, "min"],
    max = df[, "max"],
    row.names = NULL
  )
  df
}

args <- commandArgs(trailingOnly = TRUE)
output_path <- if (length(args)) args[[1]] else ""

bench <- run_assign_benchmarks()
print(bench)

if (nzchar(output_path)) {
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  write.csv(bench, output_path, row.names = FALSE)
  message("Saved benchmarks to ", output_path)
}
