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
sizes <- c(small = 500L, medium = 1000L, large = 2000L)

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
#> 1        Matrix multiply  500         base R  9.494867e+01   0.0097535105
#> 2        Matrix multiply  500            mlx  1.423622e+02   0.0030594405
#> 3        Matrix multiply 1000         base R  8.817598e+00   0.0643497050
#> 4        Matrix multiply 1000            mlx  1.177222e+02   0.0080643310
#> 5        Matrix multiply 2000         base R  3.701900e+00   0.2605896040
#> 6        Matrix multiply 2000            mlx  3.312842e+01   0.0335903570
#> 7        Matrix addition  500         base R  3.780645e+02   0.0007022890
#> 8        Matrix addition  500            mlx  2.902928e+02   0.0027942525
#> 9        Matrix addition 1000         base R  2.474537e+02   0.0022273660
#> 10       Matrix addition 1000            mlx  9.178330e+01   0.0059512320
#> 11       Matrix addition 2000         base R  7.001641e+01   0.0100836630
#> 12       Matrix addition 2000            mlx  4.674550e+01   0.0192946000
#> 13  Subset rows (vector)  500         base R  5.315200e+02   0.0012371750
#> 14  Subset rows (vector)  500            mlx  3.296629e+02   0.0026920190
#> 15  Subset rows (vector) 1000         base R  8.205221e+01   0.0091685020
#> 16  Subset rows (vector) 1000            mlx  1.710031e+02   0.0052067130
#> 17  Subset rows (vector) 2000         base R  6.235870e+01   0.0153794485
#> 18  Subset rows (vector) 2000            mlx  5.676906e+01   0.0148429430
#> 19 Subset (matrix index)  500         base R  5.410022e+04   0.0000052480
#> 20 Subset (matrix index)  500            mlx  3.333498e+02   0.0028725830
#> 21 Subset (matrix index) 1000         base R  2.772636e+04   0.0000149035
#> 22 Subset (matrix index) 1000            mlx  1.872470e+02   0.0046373460
#> 23 Subset (matrix index) 2000         base R  1.840066e+04   0.0000263630
#> 24 Subset (matrix index) 2000            mlx  6.712643e+01   0.0150386770
#> 25                   Sum  500         base R  2.922386e+03   0.0002588740
#> 26                   Sum  500            mlx  4.947150e+02   0.0017218975
#> 27                   Sum 1000         base R  7.177348e+02   0.0012599915
#> 28                   Sum 1000            mlx  6.139754e+02   0.0014713670
#> 29                   Sum 2000         base R  1.791110e+02   0.0054326230
#> 30                   Sum 2000            mlx  4.412487e+02   0.0020584870
#> 31                  Mean  500         base R  1.498735e+03   0.0005577230
#> 32                  Mean  500            mlx  5.781744e+02   0.0014643150
#> 33                  Mean 1000         base R  3.451389e+02   0.0023665610
#> 34                  Mean 1000            mlx  5.974852e+02   0.0012993515
#> 35                  Mean 2000         base R  9.976299e+01   0.0093361510
#> 36                  Mean 2000            mlx  4.163959e+02   0.0019928665
#> 37              Row sums  500         base R  9.806927e+03   0.0000567850
#> 38              Row sums  500            mlx  6.390475e+02   0.0013582890
#> 39              Row sums 1000         base R  2.615620e+03   0.0002318140
#> 40              Row sums 1000            mlx  2.663172e+02   0.0029305160
#> 41              Row sums 2000         base R  6.535350e+02   0.0011913985
#> 42              Row sums 2000            mlx  4.382283e+02   0.0021271005
#> 43             Row means  500         base R  1.001586e+04   0.0000489950
#> 44             Row means  500            mlx  3.793056e+02   0.0019803615
#> 45             Row means 1000         base R  3.174500e+03   0.0002252130
#> 46             Row means 1000            mlx  5.098386e+02   0.0017060715
#> 47             Row means 2000         base R  6.544005e+02   0.0012452520
#> 48             Row means 2000            mlx  3.755038e+02   0.0023230600
#> 49            tcrossprod  500         base R  1.498499e+02   0.0054885470
#> 50            tcrossprod  500            mlx  1.426792e+02   0.0064050815
#> 51            tcrossprod 1000         base R  3.473693e+01   0.0282496560
#> 52            tcrossprod 1000            mlx  1.355883e+02   0.0063232045
#> 53            tcrossprod 2000         base R  5.094326e+00   0.2023830110
#> 54            tcrossprod 2000            mlx  3.102551e+01   0.0324478100
#> 55               scale()  500         base R  4.589542e+01   0.0204583030
#> 56               scale()  500            mlx  1.398841e+02   0.0060722230
#> 57               scale() 1000         base R  7.298409e+00   0.0959254860
#> 58               scale() 1000            mlx  1.077013e+02   0.0103841110
#> 59               scale() 2000         base R  2.073113e+00   0.3803513830
#> 60               scale() 2000            mlx  3.837239e+01   0.0311728125
#> 61          Solve Ax = b  500         base R  7.989780e+01   0.0094741160
#> 62          Solve Ax = b  500            mlx  3.889600e+01   0.0198829295
#> 63          Solve Ax = b 1000         base R  1.892436e+01   0.0545543130
#> 64          Solve Ax = b 1000            mlx  1.296170e+01   0.0723606130
#> 65          Solve Ax = b 2000         base R  4.582469e+00   0.2060834660
#> 66          Solve Ax = b 2000            mlx  1.762571e+00   0.5218167170
#> 67             Backsolve  500         base R  6.346025e+03   0.0000619920
#> 68             Backsolve  500            mlx  1.991571e+02   0.0047954010
#> 69             Backsolve 1000         base R  2.477324e+03   0.0002168080
#> 70             Backsolve 1000            mlx  5.146298e+01   0.0174178250
#> 71             Backsolve 2000         base R  4.665327e+02   0.0016527510
#> 72             Backsolve 2000            mlx  5.523567e+00   0.1601998330
#> 73              Cholesky  500         base R  1.692913e+02   0.0046385350
#> 74              Cholesky  500            mlx  3.795302e+02   0.0022416340
#> 75              Cholesky 1000         base R  3.812578e+01   0.0238394910
#> 76              Cholesky 1000            mlx  6.955200e+01   0.0130045645
#> 77              Cholesky 2000         base R  7.973647e+00   0.1154000350
#> 78              Cholesky 2000            mlx  1.923624e+01   0.0519137900
#> 79              chol2inv  500         base R  1.461029e+02   0.0064997710
#> 80              chol2inv  500            mlx  1.715438e+02   0.0055418060
#> 81              chol2inv 1000         base R  1.783553e+01   0.0463723735
#> 82              chol2inv 1000            mlx  2.553435e+01   0.0374465300
#> 83              chol2inv 2000         base R  2.311108e+00   0.4275705500
#> 84              chol2inv 2000            mlx  3.514314e+00   0.2873535840
#> 85     SVD (values only)  500         base R  1.582794e+01   0.0626007065
#> 86     SVD (values only)  500            mlx  2.891464e+01   0.0345955950
#> 87     SVD (values only) 1000         base R  2.006351e+00   0.4984173200
#> 88     SVD (values only) 1000            mlx  4.790124e+00   0.2087628570
#> 89     SVD (values only) 2000         base R  2.761098e-01   3.6217479970
#> 90     SVD (values only) 2000            mlx  5.433025e-01   1.8405952470
#> 91              Diagonal  500         base R  2.756986e+04   0.0000168920
#> 92              Diagonal  500            mlx  4.533901e+03   0.0001468620
#> 93              Diagonal 1000         base R  3.798295e+04   0.0000123410
#> 94              Diagonal 1000            mlx  4.724615e+03   0.0000882730
#> 95              Diagonal 2000         base R  2.225496e+04   0.0000196800
#> 96              Diagonal 2000            mlx  4.175547e+03   0.0001364890
```
