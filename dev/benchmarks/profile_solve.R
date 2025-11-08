#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Rmlx)
  library(bench)
})

n <- as.integer(Sys.getenv("SOLVE_BENCH_N", "1000"))
set.seed(123)
a_r <- matrix(rnorm(n * n), n, n)
b_r <- matrix(rnorm(n), n, 1)

a <- as_mlx(a_r, dtype = "float32")
b <- as_mlx(b_r, dtype = "float32")

force_eval <- function(x) { mlx_eval(x); invisible(NULL) }

res <- bench::mark(
  inv = force_eval(solve(a)),
  solve = force_eval(solve(a, b)),
  iterations = as.integer(Sys.getenv("SOLVE_BENCH_ITER", "5")),
  check = FALSE
)
print(res)
