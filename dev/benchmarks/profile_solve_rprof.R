#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(Rmlx))
set.seed(456)
n <- as.integer(Sys.getenv("SOLVE_BENCH_N", "2000"))
a_r <- matrix(rnorm(n * n), n, n)
b_r <- matrix(rnorm(n), n, 1)

a <- as_mlx(a_r, dtype = "float32")
b <- as_mlx(b_r, dtype = "float32")

force_eval <- function(expr) {
  res <- expr
  mlx_eval(res)
  invisible(NULL)
}

profile_once <- function(expr, file) {
  Rprof(file, interval = 0.001)
  force_eval(expr)
  Rprof(NULL)
  summaryRprof(file)$by.self
}

print("Profiling inv")
print(profile_once(solve(a), "solve_inv.out"))

print("Profiling solve")
print(profile_once(solve(a, b), "solve_rhs.out"))
