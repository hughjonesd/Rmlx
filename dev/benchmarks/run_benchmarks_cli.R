#!/usr/bin/env Rscript

`%||%` <- function(x, y) if (!is.null(x)) x else y

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
source(file.path(repo_root, "dev", "benchmarks", "bench_helpers.R"))

default_sizes <- c(small = 500L, medium = 1000L, large = 2000L)

args <- commandArgs(trailingOnly = TRUE)
output_path <- file.path(repo_root, "bench_results.tsv")
sizes_arg <- NULL

i <- 1L
while (i <= length(args)) {
  arg <- args[i]
  if (arg == "--output") {
    if (i == length(args)) stop("--output requires a value")
    i <- i + 1L
    output_path <- args[i]
  } else if (startsWith(arg, "--output=")) {
    output_path <- substring(arg, 10)
  } else if (arg == "--sizes") {
    if (i == length(args)) stop("--sizes requires a value")
    i <- i + 1L
    sizes_arg <- args[i]
  } else if (startsWith(arg, "--sizes=")) {
    sizes_arg <- substring(arg, 9)
  } else {
    stop("Unknown argument: ", arg)
  }
  i <- i + 1L
}

size_vals <- if (is.null(sizes_arg)) {
  default_sizes
} else {
  vals <- as.integer(strsplit(sizes_arg, ",", fixed = TRUE)[[1]])
  vals <- vals[!is.na(vals) & vals > 0]
  if (!length(vals)) {
    stop("Invalid sizes: ", sizes_arg)
  }
  vals
}

sizes <- size_vals
names(sizes) <- if (!is.null(names(size_vals)) && all(names(size_vals) != "")) {
  names(size_vals)
} else {
  sprintf("n%d", sizes)
}

inputs <- build_benchmark_inputs(sizes)
operations <- benchmark_operations()
results <- run_benchmarks(operations, inputs)

results$median_ms <- as.numeric(results$median, units = "seconds") * 1000
backends <- c("base", "mlx")
col_keys <- as.vector(t(outer(names(sizes), backends, paste, sep = "_")))
col_headers <- paste0(col_keys, "_ms")

summary <- data.frame(
  operation = vapply(operations, `[[`, character(1), "label"),
  matrix(NA_real_, nrow = length(operations), ncol = length(col_headers)),
  stringsAsFactors = FALSE
)
names(summary)[-1] <- col_headers

for (i in seq_len(nrow(results))) {
  op <- results$operation[i]
  key <- paste(results$size[i], results$backend[i], sep = "_")
  col <- paste0(key, "_ms")
  summary[summary$operation == op, col] <- results$median_ms[i]
}

format_val <- function(x) {
  out <- rep("", length(x))
  keep <- !is.na(x)
  out[keep] <- sprintf("%.4f", x[keep])
  out
}
summary[-1] <- lapply(summary[-1], format_val)

write_tsv <- function(df, path) {
  con <- if (is.null(path)) stdout() else file(path, open = "w")
  on.exit(if (!is.null(path)) close(con), add = TRUE)
  header <- paste(names(df), collapse = "\t")
  writeLines(header, con)
  apply(df, 1, function(row) writeLines(paste(row, collapse = "\t"), con))
}

write_tsv(summary, output_path)
message("Benchmark results written to ", output_path, " (", nrow(results), " measurements).")
