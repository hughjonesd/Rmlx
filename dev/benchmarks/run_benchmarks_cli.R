#!/usr/bin/env Rscript

args_full <- commandArgs(FALSE)
script_dir <- {
  file_arg <- grep("^--file=", args_full, value = TRUE)
  if (length(file_arg)) {
    dirname(sub("^--file=", "", file_arg[1]))
  } else {
    getwd()
  }
}
repo_root <- normalizePath(file.path(script_dir, "..", ".."), mustWork = TRUE)
suppressPackageStartupMessages(
  source(file.path(repo_root, "dev", "benchmarks", "bench_helpers.R"))
)

sizes <- c(n500 = 500L, n1000 = 1000L, n2000 = 2000L)
inputs <- build_benchmark_inputs(sizes)

ops <- list(
  list(
    label = "matmul",
    fn = function(input) force_mlx(input$mlx$a %*% input$mlx$b)
  ),
  list(
    label = "solve",
    fn = function(input) force_mlx(solve(input$mlx$spd, input$mlx$rhs))
  ),
  list(
    label = "as_mlx",
    fn = function(input) force_mlx(as_mlx(input$base$a, dtype = "float32"))
  )
)

min_time <- 0.05
min_iter <- 2L

results <- matrix(
  NA_real_,
  nrow = length(ops),
  ncol = length(sizes),
  dimnames = list(vapply(ops, `[[`, character(1), "label"), names(sizes))
)

for (size_name in names(sizes)) {
  data <- inputs[[size_name]]
  for (op in ops) {
    bench_res <- bench::mark(
      op$fn(data),
      min_time = min_time,
      min_iterations = min_iter,
      check = FALSE,
      filter_gc = FALSE
    )
    results[op$label, size_name] <- as.numeric(bench_res$median, units = "secs") * 1000
  }
}

df <- data.frame(
  op = rownames(results),
  results,
  row.names = NULL,
  check.names = FALSE
)
df[-1] <- lapply(df[-1], function(x) sprintf("%.4f", x))

cat("Median milliseconds per operation\n")
print(df, row.names = FALSE)
