#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("bench", quietly = TRUE)) {
    stop("The bench package is required. Install it with install.packages('bench').", call. = FALSE)
  }
  if (!requireNamespace("pkgload", quietly = TRUE)) {
    stop("The pkgload package is required. Install it with install.packages('pkgload').", call. = FALSE)
  }
})

# Load the package code from the current checkout so we benchmark the branch in use.
pkgload::load_all(export_all = FALSE, helpers = FALSE, quiet = TRUE)

args <- commandArgs(trailingOnly = TRUE)

default_size <- 4000L
default_iterations <- 5L
default_device <- Sys.getenv("AS_MLX_BENCH_DEVICE", unset = "cpu")

matrix_size <- if (length(args) >= 1L) as.integer(args[[1L]]) else default_size
if (is.na(matrix_size) || matrix_size <= 0L) {
  stop("Matrix size must be a positive integer (number of rows for a square matrix).", call. = FALSE)
}

iterations <- if (length(args) >= 2L) as.integer(args[[2L]]) else default_iterations
if (is.na(iterations) || iterations <= 0L) {
  stop("Iterations must be a positive integer.", call. = FALSE)
}

device <- if (length(args) >= 3L) args[[3L]] else default_device

message("Preparing benchmark data (", matrix_size, " x ", matrix_size, " numeric matrix)...")
set.seed(123)
matrix_payload <- matrix(runif(matrix_size * matrix_size), nrow = matrix_size)

current_branch <- system("git rev-parse --abbrev-ref HEAD", intern = TRUE)
current_commit <- system("git rev-parse --short HEAD", intern = TRUE)

bench_res <- bench::mark(
  as_mlx(matrix_payload, device = device),
  iterations = iterations,
  check = FALSE,
  time_unit = "s",
  memory = TRUE,
  min_time = 0
)

print(bench_res)

bench_df <- as.data.frame(bench_res)
names(bench_df) <- gsub("/", "_per_", names(bench_df), fixed = TRUE)
bench_df$branch <- trimws(current_branch)
bench_df$commit <- trimws(current_commit)
bench_df$timestamp <- format(Sys.time(), tz = "UTC", usetz = TRUE)
bench_df$rows <- matrix_size
bench_df$cols <- matrix_size
bench_df$device <- device

output_dir <- file.path("dev", "benchmarks")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

output_path <- file.path(output_dir, "as_mlx_branch_benchmarks.csv")

col_order <- c("timestamp", "branch", "commit", "rows", "cols", "device",
               "expression", "min", "median", "itr_per_sec", "mem_alloc", "gc_per_sec", "n_gc")
bench_df <- bench_df[, intersect(col_order, names(bench_df)), drop = FALSE]

write_header <- !file.exists(output_path)
utils::write.table(
  bench_df,
  file = output_path,
  sep = ",",
  row.names = FALSE,
  col.names = write_header,
  append = !write_header
)

message("Benchmark results appended to ", output_path)
