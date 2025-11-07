# MLX Benchmarks

This vignette compares base R and MLX timings across core matrix
routines.

``` r
library(Rmlx)
#> 
#> Attaching package: 'Rmlx'
#> The following object is masked from 'package:stats':
#> 
#>     fft
#> The following objects are masked from 'package:base':
#> 
#>     asplit, backsolve, chol2inv, col, colMeans, colSums, diag, drop,
#>     outer, row, rowMeans, rowSums, svd
library(bench)
library(ggplot2)
```

``` r
sizes <- c(small = 1000L, medium = 2000L, large = 4000L)

set.seed(20251031)

`%||%` <- function(x, y) if (!is.null(x)) x else y

force_mlx <- function(x) {
  if (inherits(x, "mlx")) {
    mlx_eval(x)
  } else if (is.list(x)) {
    lapply(x, force_mlx)
  }
  invisible(NULL)
}

inputs <- lapply(sizes, function(n) {
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

  list(
    base = base_data,
    mlx = mlx_data
  )
})

operations <- list(
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

default_min_time <- 0.25
default_min_iterations <- 3L

run_benchmarks <- function(ops, inputs) {
  rows <- vector("list", length(ops) * length(inputs))
  idx <- 1L
  for (op in ops) {
    for (size_name in names(inputs)) {
      base_data <- inputs[[size_name]]$base
      mlx_data <- inputs[[size_name]]$mlx
      bench_res <- bench::mark(
        base = op$base(base_data),
        mlx = op$mlx(mlx_data),
        min_time = op$min_time %||% default_min_time,
        min_iterations = op$min_iterations %||% default_min_iterations,
        check = FALSE,
        filter_gc = FALSE
      )
      rows[[idx]] <- data.frame(
        operation = op$label,
        size = size_name,
        implementation = as.character(bench_res$expression),
        iters_per_sec = as.numeric(bench_res[["itr/sec"]]),
        median_seconds = as.numeric(bench_res$median, units = "seconds"),
        stringsAsFactors = FALSE
      )
      idx <- idx + 1L
    }
  }
  do.call(rbind, rows)
}

bench_results <- run_benchmarks(operations, inputs)

bench_results$size <- factor(
  bench_results$size,
  levels = names(sizes),
  labels = sizes
)
bench_results$implementation <- factor(
  bench_results$implementation,
  levels = c("base", "mlx"),
  labels = c("base R", "mlx")
)
bench_results$operation <- factor(
  bench_results$operation,
  levels = vapply(operations, `[[`, character(1), "label")
)
```

``` r
ggplot(
  bench_results,
  aes(x = size, y = median_seconds, colour = implementation, group = implementation)
) +
  geom_line() +
  geom_point(size = 2) +
  scale_colour_manual(values = c("base R" = "#4A4A4A", "mlx" = "#D63230")) +
  facet_wrap(~ operation, scales = "free_y") +
  labs(
    title = "Benchmarks for common matrix operations",
    subtitle = "Time taken (less is better)",
    x = "Matrix size",
    y = "Median time (seconds)",
    colour = ""
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")
```

![](benchmarks_files/figure-html/plot-1.png)

``` r
bench_results
#>                operation size implementation iters_per_sec median_seconds
#> 1        Matrix multiply 1000         base R  1.686332e+01   5.337489e-02
#> 2        Matrix multiply 1000            mlx  1.482274e+02   5.776613e-03
#> 3        Matrix multiply 2000         base R  2.694053e+00   3.847019e-01
#> 4        Matrix multiply 2000            mlx  2.063118e+01   3.706332e-02
#> 5        Matrix multiply 4000         base R  6.928183e-01   1.339366e+00
#> 6        Matrix multiply 4000            mlx  6.535483e+00   1.216408e-01
#> 7        Matrix addition 1000         base R  7.580863e+01   1.253397e-02
#> 8        Matrix addition 1000            mlx  2.069565e+02   4.067672e-03
#> 9        Matrix addition 2000         base R  3.445317e+01   1.812093e-02
#> 10       Matrix addition 2000            mlx  6.703484e+01   1.198967e-02
#> 11       Matrix addition 4000         base R  1.575538e+01   4.288723e-02
#> 12       Matrix addition 4000            mlx  2.023367e+01   2.637891e-02
#> 13  Subset rows (vector) 1000         base R  9.199962e+01   6.712602e-03
#> 14  Subset rows (vector) 1000            mlx  2.258265e+02   4.155842e-03
#> 15  Subset rows (vector) 2000         base R  3.579691e+01   9.720444e-03
#> 16  Subset rows (vector) 2000            mlx  1.116736e+02   7.904595e-03
#> 17  Subset rows (vector) 4000         base R  1.852686e+01   5.096275e-02
#> 18  Subset rows (vector) 4000            mlx  2.542716e+01   2.937162e-02
#> 19 Subset (matrix index) 1000         base R  3.077650e+04   1.414500e-05
#> 20 Subset (matrix index) 1000            mlx  7.057360e+02   1.069977e-03
#> 21 Subset (matrix index) 2000         base R  1.060398e+04   3.478850e-05
#> 22 Subset (matrix index) 2000            mlx  1.226078e+02   4.105269e-03
#> 23 Subset (matrix index) 4000         base R  3.627776e+03   8.909300e-05
#> 24 Subset (matrix index) 4000            mlx  2.803131e+02   3.321861e-03
#> 25                   Sum 1000         base R  8.158334e+02   1.025717e-03
#> 26                   Sum 1000            mlx  5.693211e+02   1.403553e-03
#> 27                   Sum 2000         base R  1.990504e+02   4.471522e-03
#> 28                   Sum 2000            mlx  4.187205e+02   1.885180e-03
#> 29                   Sum 4000         base R  5.644120e+01   1.740512e-02
#> 30                   Sum 4000            mlx  3.343564e+02   2.706328e-03
#> 31                  Mean 1000         base R  3.827968e+02   2.436958e-03
#> 32                  Mean 1000            mlx  6.183002e+02   1.443118e-03
#> 33                  Mean 2000         base R  1.189897e+02   8.147951e-03
#> 34                  Mean 2000            mlx  5.898661e+02   1.595228e-03
#> 35                  Mean 4000         base R  2.426965e+01   4.116203e-02
#> 36                  Mean 4000            mlx  3.177675e+02   2.935477e-03
#> 37              Row sums 1000         base R  4.508088e+03   1.935200e-04
#> 38              Row sums 1000            mlx  6.190800e+02   1.350048e-03
#> 39              Row sums 2000         base R  1.101625e+03   8.162895e-04
#> 40              Row sums 2000            mlx  8.397160e+02   9.849020e-04
#> 41              Row sums 4000         base R  2.619458e+02   3.322353e-03
#> 42              Row sums 4000            mlx  3.497414e+02   2.578695e-03
#> 43             Row means 1000         base R  4.561999e+03   1.928230e-04
#> 44             Row means 1000            mlx  1.117836e+03   6.942325e-04
#> 45             Row means 2000         base R  1.282473e+03   7.581310e-04
#> 46             Row means 2000            mlx  9.342403e+02   1.043471e-03
#> 47             Row means 4000         base R  3.380841e+02   2.990089e-03
#> 48             Row means 4000            mlx  5.398472e+02   1.758613e-03
#> 49            tcrossprod 1000         base R  4.248881e+01   1.443985e-02
#> 50            tcrossprod 1000            mlx  1.151698e+02   4.997777e-03
#> 51            tcrossprod 2000         base R  5.960182e+00   1.865696e-01
#> 52            tcrossprod 2000            mlx  5.226876e+01   1.909452e-02
#> 53            tcrossprod 4000         base R  1.378266e+00   7.290075e-01
#> 54            tcrossprod 4000            mlx  8.418419e+00   1.142077e-01
#> 55               scale() 1000         base R  8.506913e+00   1.028171e-01
#> 56               scale() 1000            mlx  3.992617e+01   1.817208e-02
#> 57               scale() 2000         base R  6.827832e+00   1.393218e-01
#> 58               scale() 2000            mlx  2.200685e+01   4.338149e-02
#> 59               scale() 4000         base R  5.892145e-01   1.258455e+00
#> 60               scale() 4000            mlx  4.959993e+00   2.078249e-01
#> 61          Solve Ax = b 1000         base R  4.647989e+01   2.071650e-02
#> 62          Solve Ax = b 1000            mlx  2.420378e+01   3.842868e-02
#> 63          Solve Ax = b 2000         base R  6.384171e+00   1.466064e-01
#> 64          Solve Ax = b 2000            mlx  4.225594e+00   2.363753e-01
#> 65          Solve Ax = b 4000         base R  9.146701e-01   1.071055e+00
#> 66          Solve Ax = b 4000            mlx  3.200219e-01   3.158153e+00
#> 67             Backsolve 1000         base R  5.173912e+03   1.873495e-04
#> 68             Backsolve 1000            mlx  6.759129e+01   1.230894e-02
#> 69             Backsolve 2000         base R  5.550587e+02   1.405193e-03
#> 70             Backsolve 2000            mlx  7.332984e+00   1.372084e-01
#> 71             Backsolve 4000         base R  1.432211e+02   6.272098e-03
#> 72             Backsolve 4000            mlx  7.410269e-01   1.397919e+00
#> 73              Cholesky 1000         base R  4.237244e+01   2.003383e-02
#> 74              Cholesky 1000            mlx  1.319401e+02   6.263611e-03
#> 75              Cholesky 2000         base R  9.359651e+00   1.115690e-01
#> 76              Cholesky 2000            mlx  3.516983e+01   2.553537e-02
#> 77              Cholesky 4000         base R  2.735014e+00   3.630021e-01
#> 78              Cholesky 4000            mlx  7.707886e+00   1.274542e-01
#> 79              chol2inv 1000         base R  4.254603e+01   2.240232e-02
#> 80              chol2inv 1000            mlx  5.800550e+01   1.695321e-02
#> 81              chol2inv 2000         base R  4.408622e+00   2.249110e-01
#> 82              chol2inv 2000            mlx  7.933741e+00   1.099304e-01
#> 83              chol2inv 4000         base R  3.642869e-01   2.692015e+00
#> 84              chol2inv 4000            mlx  7.427381e-01   1.336821e+00
#> 85     SVD (values only) 1000         base R  4.701149e+00   2.127139e-01
#> 86     SVD (values only) 1000            mlx  8.991039e+00   1.093591e-01
#> 87     SVD (values only) 2000         base R  4.985447e-01   2.005838e+00
#> 88     SVD (values only) 2000            mlx  1.487637e+00   6.722072e-01
#> 89     SVD (values only) 4000         base R  4.754891e-02   2.103098e+01
#> 90     SVD (values only) 4000            mlx  1.149650e-01   8.698301e+00
#> 91              Diagonal 1000         base R  1.798698e+04   1.750700e-05
#> 92              Diagonal 1000            mlx  8.222750e+03   9.913800e-05
#> 93              Diagonal 2000         base R  1.058169e+04   4.358300e-05
#> 94              Diagonal 2000            mlx  6.713126e+03   9.712900e-05
#> 95              Diagonal 4000         base R  1.414439e+04   5.781000e-05
#> 96              Diagonal 4000            mlx  6.132823e+03   1.389900e-04
```
