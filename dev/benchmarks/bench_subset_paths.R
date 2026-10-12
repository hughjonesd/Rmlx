#!/usr/bin/env Rscript
# Run on the baseline and candidate with an output CSV path as the first argument.
# Inputs are evaluated before timing; every result is evaluated inside the timing.
suppressPackageStartupMessages(devtools::load_all(quiet = TRUE))
suppressPackageStartupMessages(source("inst/benchmarks/bench_helpers.R"))

set.seed(20261012)
mat <- matrix(runif(1000 * 1000), 1000)
arr <- array(runif(60 * 70 * 80), c(60, 70, 80))
rows <- 201:800
cols <- 101:900
rows_irregular <- sample(1000, 600)
cols_irregular <- sample(1000, 800)
mask <- rep(c(TRUE, FALSE), 500)
cases <- list(
  matrix_full = quote(x[, ]),
  matrix_contiguous = quote(x[rows, cols]),
  matrix_rows = quote(x[rows, ]),
  matrix_irregular = quote(x[rows_irregular, cols_irregular]),
  matrix_mlx_indices = quote(x[rows_mlx, cols_mlx]),
  matrix_logical = quote(x[mask, mask]),
  array_full = quote(y[, , ]),
  array_contiguous = quote(y[11:50, 11:60, 11:70]),
  assign_contiguous = quote({ z <- x; z[rows, cols] <- values; z }),
  assign_scalar = quote({ z <- x; z[rows, cols] <- 0; z }),
  assign_irregular = quote({ z <- x; z[rows_irregular, cols_irregular] <- values; z }),
  assign_mlx_indices = quote({ z <- x; z[rows_mlx, cols_mlx] <- values; z })
)
results <- list()
for (device in c("gpu", "cpu")) {
  mlx_device(device)
  x <- as_mlx(mat)
  y <- as_mlx(arr)
  rows_mlx <- as_mlx(rows, dtype = "int32")
  cols_mlx <- as_mlx(cols, dtype = "int32")
  values <- as_mlx(matrix(runif(length(rows) * length(cols)), length(rows)))
  force_mlx(list(x, y, rows_mlx, cols_mlx, values))
  for (scenario in names(cases)) {
    expr <- cases[[scenario]]
    for (warmup in 1:5) force_mlx(eval(expr))
    timing <- bench::mark(
      force_mlx(eval(expr)), iterations = 50, check = FALSE,
      memory = FALSE, filter_gc = FALSE
    )
    results[[length(results) + 1L]] <- data.frame(
      device = device, scenario = scenario,
      median_ms = as.numeric(timing$median) * 1000
    )
  }
}
results <- do.call(rbind, results)
print(results, row.names = FALSE)
args <- commandArgs(trailingOnly = TRUE)
if (length(args)) write.csv(results, args[[1]], row.names = FALSE)
