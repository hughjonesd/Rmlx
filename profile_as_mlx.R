#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(devtools)
})

devtools::load_all(quiet = TRUE)

dtype <- "float32"
n <- as.integer(Sys.getenv("AS_MLX_N", "1000"))
m <- as.integer(Sys.getenv("AS_MLX_M", as.character(n)))
iters <- as.integer(Sys.getenv("AS_MLX_ITERS", "20"))
iters_breakdown <- as.integer(Sys.getenv("AS_MLX_BREAKDOWN_ITERS", "10"))
bench_iters <- as.integer(Sys.getenv("AS_MLX_BENCH_ITERS", "5"))

device <- mlx_default_device()
cat(sprintf("Profiling as_mlx() on %dx%d matrix (%s, dtype=%s)\n", n, m, device, dtype))

payload <- matrix(runif(n * m), nrow = n)

force_eval <- function(arr) {
  mlx_eval(arr)
  invisible(NULL)
}

cat("Warm-up...\n")
force_eval(as_mlx(payload, dtype = dtype))

mean_time <- system.time({
  for (i in seq_len(iters)) {
    arr <- as_mlx(payload, dtype = dtype)
    force_eval(arr)
  }
})["elapsed"] / iters

cat(sprintf("Mean elapsed per as_mlx() call (with mlx_eval): %.6f sec\n", mean_time))

prof_file <- "profile_as_mlx_rprof.out"
cat(sprintf("Collecting Rprof samples (%d iterations, interval=0.5ms)...\n", iters))
Rprof(prof_file, interval = 0.0005)
for (i in seq_len(iters)) {
  arr <- as_mlx(payload, dtype = dtype)
  force_eval(arr)
}
Rprof(NULL)
cat(sprintf("Saved Rprof data to %s\n", prof_file))

profile <- summaryRprof(prof_file)
print(head(profile$by.self, 10))

measure <- function(fn, iter = iters_breakdown) {
  gc(FALSE)
  elapsed <- system.time({
    for (i in seq_len(iter)) {
      fn()
    }
  })["elapsed"]
  as.numeric(elapsed) / iter
}

cat("\nBreakdown (per-call means over", iters_breakdown, "iterations):\n")

payload_time <- measure(function() {
  invisible(Rmlx:::.mlx_coerce_payload(payload, dtype))
})
cat(sprintf("  .mlx_coerce_payload(): %.6f sec\n", payload_time))

dims <- Rmlx:::.mlx_infer_dim(payload)
payload_vec <- Rmlx:::.mlx_coerce_payload(payload, dtype)
handle <- Rmlx:::.mlx_resolve_device(device, mlx_default_device())

cpp_time <- measure(function() {
  ptr <- Rmlx:::.mlx_eval_with_stream(handle, function(dev) {
    Rmlx:::cpp_mlx_from_r(payload_vec, as.integer(dims), dtype, dev)
  })
  Rmlx:::cpp_mlx_eval(ptr)
  invisible(NULL)
})
cat(sprintf("  cpp_mlx_from_r() + eval(): %.6f sec\n", cpp_time))

if (requireNamespace("bench", quietly = TRUE)) {
  cat(sprintf("\nbench::mark (iterations=%d, check=FALSE):\n", bench_iters))
  bench_res <- bench::mark(
    as_mlx = {
      arr <- as_mlx(payload, dtype = dtype)
      force_eval(arr)
      invisible(NULL)
    },
    iterations = bench_iters,
    check = FALSE
  )
  print(bench_res[, c("expression", "median", "itr/sec", "mem_alloc")])
} else {
  cat("\nbench package not installed; skipping bench::mark().\n")
}

cat("\nDone.\n")
