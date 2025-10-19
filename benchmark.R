#!/usr/bin/env Rscript
# Benchmark: Matrix Multiplication in Base R vs Rmlx (GPU)
#
# This script compares the performance of matrix multiplication
# between base R and Rmlx on Apple Silicon GPU.
#
# Usage: Rscript benchmark.R

library(Rmlx)
library(bench)

cat("\n")
cat("=================================================================\n")
cat("   Matrix Multiplication Benchmark: Base R vs Rmlx (GPU)\n")
cat("=================================================================\n\n")

# Test different matrix sizes
results <- data.frame(
  Size = character(),
  Base_R = character(),
  Rmlx_GPU = character(),
  Speedup = numeric(),
  stringsAsFactors = FALSE
)

sizes <- c(500, 1000, 2000)

for (n in sizes) {
  cat("Testing", n, "x", n, "matrices...\n")

  # Create test matrices
  set.seed(123)
  m1 <- matrix(rnorm(n * n), n, n)
  m2 <- matrix(rnorm(n * n), n, n)

  # Convert to MLX (float32)
  m1_mlx <- as_mlx(m1, dtype = "float32")
  m2_mlx <- as_mlx(m2, dtype = "float32")

  # Benchmark
  bm <- bench::mark(
    base_r = {
      result <- m1 %*% m2
    },
    rmlx_gpu = {
      result_mlx <- m1_mlx %*% m2_mlx
      mlx_eval(result_mlx)
    },
    check = FALSE,
    iterations = 10,
    filter_gc = FALSE
  )

  speedup <- as.numeric(bm$median[1] / bm$median[2])

  results <- rbind(results, data.frame(
    Size = paste0(n, " x ", n),
    Base_R = format(bm$median[1]),
    Rmlx_GPU = format(bm$median[2]),
    Speedup = round(speedup, 1)
  ))
}

cat("\n")
print(results, row.names = FALSE)

cat("\n=================================================================\n")
cat("Note: Rmlx uses float32 on Apple Silicon GPU (M-series chip)\n")
cat("      Base R uses float64 (double precision)\n")
cat("=================================================================\n\n")
