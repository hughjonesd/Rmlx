suppressPackageStartupMessages({
  library(bench)
  library(devtools)
})

devtools::load_all(".", quiet = TRUE)

sizes <- as.integer(strsplit(
  Sys.getenv("RMLX_QR_GPU_SIZES", "100000,1000000"),
  ",",
  fixed = TRUE
)[[1L]])
iterations <- as.integer(Sys.getenv("RMLX_QR_GPU_ITERATIONS", "3"))
p <- as.integer(Sys.getenv("RMLX_QR_GPU_P", "50"))
block_rows <- as.integer(Sys.getenv("RMLX_QR_GPU_BLOCK_ROWS", "2048"))
include_metal_householder <- identical(
  Sys.getenv("RMLX_QR_GPU_METAL_HOUSEHOLDER", "false"), "true"
)
include_householder <- identical(
  Sys.getenv("RMLX_QR_GPU_HOUSEHOLDER", "false"), "true"
)
include_blocked_householder <- identical(
  Sys.getenv("RMLX_QR_GPU_BLOCKED_HOUSEHOLDER", "false"), "true"
)
include_tsqr <- identical(Sys.getenv("RMLX_QR_GPU_TSQR", "false"), "true")

set.seed(20260613)

run_one <- function(n) {
  x <- as_mlx(matrix(rnorm(n * p), n, p))
  y <- as_mlx(matrix(rnorm(n), n, 1))

  cpu_qr <- function() {
    q <- qr(x, device = "cpu")
    qty <- with_device("cpu", crossprod(q$Q, y))
    coef <- mlx_solve_triangular(q$R, qty, upper = TRUE, device = "cpu")
    mlx_eval(coef)
    invisible(NULL)
  }

  gpu_qr <- function() {
    q <- mlx_qr_gpu(x, y)
    coef <- mlx_solve_triangular(q$R, q$qty, upper = TRUE, device = "cpu")
    mlx_eval(coef)
    invisible(NULL)
  }

  q <- mlx_qr_gpu(x, y)
  mlx_eval(q$R)

  if (include_metal_householder || include_blocked_householder ||
      include_householder || include_tsqr) {
    exprs <- list(
      cpu_qr = quote(cpu_qr()),
      gpu_qr_cholqr = quote(gpu_qr())
    )

    gpu_qr_metal_householder <- function() {
      q <- mlx_qr_gpu(x, y, method = "metal_householder")
      coef <- mlx_solve_triangular(q$R, q$qty, upper = TRUE, device = "cpu")
      mlx_eval(coef)
      invisible(NULL)
    }
    if (include_metal_householder) {
      q <- mlx_qr_gpu(x, y, method = "metal_householder")
      mlx_eval(q$R)
      exprs$gpu_qr_metal_householder <- quote(gpu_qr_metal_householder())
    }

    gpu_qr_blocked_householder <- function() {
      q <- mlx_qr_gpu(x, y, method = "blocked_householder")
      coef <- mlx_solve_triangular(q$R, q$qty, upper = TRUE, device = "cpu")
      mlx_eval(coef)
      invisible(NULL)
    }
    if (include_blocked_householder) {
      q <- mlx_qr_gpu(x, y, method = "blocked_householder")
      mlx_eval(q$R)
      exprs$gpu_qr_blocked_householder <- quote(gpu_qr_blocked_householder())
    }

    gpu_qr_householder <- function() {
      q <- mlx_qr_gpu(x, y, method = "householder")
      coef <- mlx_solve_triangular(q$R, q$qty, upper = TRUE, device = "cpu")
      mlx_eval(coef)
      invisible(NULL)
    }
    if (include_householder) {
      q <- mlx_qr_gpu(x, y, method = "householder")
      mlx_eval(q$R)
      exprs$gpu_qr_householder <- quote(gpu_qr_householder())
    }

    gpu_qr_tsqr <- function() {
      q <- mlx_qr_gpu(x, y, block_rows = block_rows, method = "tsqr")
      coef <- mlx_solve_triangular(q$R, q$qty, upper = TRUE, device = "cpu")
      mlx_eval(coef)
      invisible(NULL)
    }
    if (include_tsqr) {
      q <- mlx_qr_gpu(x, y, block_rows = block_rows, method = "tsqr")
      mlx_eval(q$R)
      exprs$gpu_qr_tsqr <- quote(gpu_qr_tsqr())
    }

    result <- do.call(
      bench::mark,
      c(exprs, list(iterations = iterations, check = FALSE, memory = FALSE))
    )
  } else {
    result <- bench::mark(
      cpu_qr = cpu_qr(),
      gpu_qr_cholqr = gpu_qr(),
      iterations = iterations,
      check = FALSE,
      memory = FALSE
    )
  }
  result$n <- n
  result$p <- p
  result$block_rows <- block_rows
  result
}

results <- do.call(rbind, lapply(sizes, run_one))
print(results[, c("n", "p", "block_rows", "expression", "min", "median", "itr/sec")])
