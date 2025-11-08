#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(devtools)
  library(bench)
})

args <- commandArgs(trailingOnly = TRUE)
sizes <- if (length(args)) {
  as.integer(args)
} else {
  c(1000L, 2000L)
}

sizes <- sizes[!is.na(sizes) & sizes > 0]
if (!length(sizes)) {
  stop("No valid matrix sizes provided.")
}

load_all(quiet = TRUE)

run_case <- function(n, iterations = 50) {
  cat(sprintf("\n== %dx%d ==\n", n, n))
  vec <- runif(n * n)

  res <- bench::mark(
    as_mlx = {
      mat <- matrix(vec, nrow = n, ncol = n)
      arr <- as_mlx(mat, dtype = "float32")
      mlx_eval(arr)
      invisible(NULL)
    },
    mlx_matrix = {
      arr <- mlx_matrix(vec, nrow = n, ncol = n, dtype = "float32")
      mlx_eval(arr)
      invisible(NULL)
    },
    iterations = iterations,
    check = FALSE
  )

  print(res[, c("expression", "median", "itr/sec", "mem_alloc")])

  quants <- lapply(
    res$time,
    function(t) quantile(as.numeric(t) * 1000, probs = c(0.25, 0.5, 0.75, 0.9, 0.95))
  )
  names(quants) <- res$expression
  print(quants)
}

for (n in sizes) {
  run_case(n)
}
