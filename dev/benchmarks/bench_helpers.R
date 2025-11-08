library(Rmlx)
library(bench)

`%||%` <- function(x, y) if (!is.null(x)) x else y

force_mlx <- function(x) {
  if (inherits(x, "mlx")) {
    mlx_eval(x)
  } else if (is.list(x)) {
    lapply(x, force_mlx)
  }
  invisible(NULL)
}

build_benchmark_inputs <- function(sizes, seed = 20251031L) {
  set.seed(seed)
  lapply(sizes, function(n) {
    a <- matrix(rnorm(n * n), n, n)
    b <- matrix(rnorm(n * n), n, n)
    spd <- crossprod(a) + diag(n) * 1e-3
    rhs <- matrix(rnorm(n), n, 1)
    chol_base <- chol(spd)
    idx_vec <- sample.int(n, size = n, replace = TRUE)
    idx_mat <- cbind(
      sample.int(n, size = n, replace = TRUE),
      sample.int(n, size = n, replace = TRUE)
    )

    base_data <- list(
      a = a,
      b = b,
      spd = spd,
      rhs = rhs,
      chol = chol_base,
      idx_vec = idx_vec,
      idx_mat = idx_mat
    )
    mlx_data <- list(
      a = as_mlx(a, dtype = "float32"),
      b = as_mlx(b, dtype = "float32"),
      spd = as_mlx(spd, dtype = "float32"),
      rhs = as_mlx(rhs, dtype = "float32"),
      chol = as_mlx(chol_base, dtype = "float32"),
      idx_vec = idx_vec,
      idx_mat = idx_mat
    )
    force_mlx(mlx_data)

    list(base = base_data, mlx = mlx_data)
  })
}

default_min_time <- 0.25
default_min_iterations <- 3L

benchmark_operations <- function() {
  list(
    list(
      id = "matmul",
      label = "Matrix multiply",
      base = function(data) { data$a %*% data$b; invisible(NULL) },
      mlx = function(data) { force_mlx(data$a %*% data$b) }
    ),
    list(
      id = "add",
      label = "Matrix addition",
      base = function(data) { data$a + data$b; invisible(NULL) },
      mlx = function(data) { force_mlx(data$a + data$b) }
    ),
    list(
      id = "subset_vec",
      label = "Subset rows (vector)",
      base = function(data) { data$a[data$idx_vec, , drop = FALSE]; invisible(NULL) },
      mlx = function(data) { force_mlx(data$a[data$idx_vec, , drop = FALSE]) }
    ),
    list(
      id = "subset_mat",
      label = "Subset (matrix index)",
      base = function(data) { data$a[data$idx_mat]; invisible(NULL) },
      mlx = function(data) { force_mlx(data$a[data$idx_mat]) }
    ),
    list(
      id = "sum",
      label = "Sum",
      base = function(data) { sum(data$a); invisible(NULL) },
      mlx = function(data) { force_mlx(sum(data$a)) }
    ),
    list(
      id = "mean",
      label = "Mean",
      base = function(data) { mean(data$a); invisible(NULL) },
      mlx = function(data) { force_mlx(mean(data$a)) }
    ),
    list(
      id = "rowsums",
      label = "Row sums",
      base = function(data) { rowSums(data$a); invisible(NULL) },
      mlx = function(data) { force_mlx(rowSums(data$a)) }
    ),
    list(
      id = "rowmeans",
      label = "Row means",
      base = function(data) { rowMeans(data$a); invisible(NULL) },
      mlx = function(data) { force_mlx(rowMeans(data$a)) }
    ),
    list(
      id = "tcrossprod",
      label = "tcrossprod",
      base = function(data) { tcrossprod(data$a); invisible(NULL) },
      mlx = function(data) { force_mlx(tcrossprod(data$a)) }
    ),
    list(
      id = "scale",
      label = "scale()",
      base = function(data) { scale(data$a); invisible(NULL) },
      mlx = function(data) { force_mlx(scale(data$a)) }
    ),
    list(
      id = "solve",
      label = "Solve Ax = b",
      base = function(data) { solve(data$spd, data$rhs); invisible(NULL) },
      mlx = function(data) { force_mlx(solve(data$spd, data$rhs)) }
    ),
    list(
      id = "backsolve",
      label = "Backsolve",
      base = function(data) { backsolve(data$chol, data$rhs); invisible(NULL) },
      mlx = function(data) { force_mlx(backsolve(data$chol, data$rhs)) }
    ),
    list(
      id = "chol",
      label = "Cholesky",
      base = function(data) { chol(data$spd); invisible(NULL) },
      mlx = function(data) { force_mlx(chol(data$spd)) }
    ),
    list(
      id = "chol2inv",
      label = "chol2inv",
      base = function(data) { chol2inv(data$chol); invisible(NULL) },
      mlx = function(data) { force_mlx(chol2inv(data$chol)) }
    ),
    list(
      id = "svd",
      label = "SVD (values only)",
      min_iterations = 1L,
      base = function(data) { svd(data$a, nu = 0, nv = 0); invisible(NULL) },
      mlx = function(data) { force_mlx(svd(data$a, nu = 0, nv = 0)) }
    ),
    list(
      id = "diag",
      label = "Diagonal",
      base = function(data) { diag(data$a); invisible(NULL) },
      mlx = function(data) { force_mlx(diag(data$a)) }
    )
  )
}

run_benchmarks <- function(operations, inputs) {
  rows <- vector("list", length(operations) * length(inputs))
  idx <- 1L
  for (op in operations) {
    for (size_name in names(inputs)) {
      bench_res <- bench::mark(
        base = op$base(inputs[[size_name]]$base),
        mlx = op$mlx(inputs[[size_name]]$mlx),
        min_time = op$min_time %||% default_min_time,
        min_iterations = op$min_iterations %||% default_min_iterations,
        check = FALSE,
        filter_gc = FALSE
      )
      rows[[idx]] <- data.frame(
        operation = op$label,
        size = size_name,
        backend = as.character(bench_res$expression),
        median = bench_res$median,
        itr_sec = bench_res$`itr/sec`,
        mem_alloc = bench_res$mem_alloc
      )
      idx <- idx + 1L
    }
  }
  do.call(rbind, rows)
}
