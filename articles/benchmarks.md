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

helpers_path <- system.file("benchmarks", "bench_helpers.R", package = "Rmlx")
if (!nzchar(helpers_path)) {
  fallbacks <- c(
    file.path("inst", "benchmarks", "bench_helpers.R"),
    file.path("dev", "benchmarks", "bench_helpers.R")
  )
  helpers_path <- fallbacks[file.exists(fallbacks)][1]
}
if (is.na(helpers_path)) {
  stop("bench_helpers.R not found. Ensure the package is installed or the inst/benchmarks directory is available.")
}
source(helpers_path)
```

``` r
sizes <- c(small = 500L, medium = 1000L, large = 2000L)
inputs <- build_benchmark_inputs(sizes)
operations <- benchmark_operations()

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
#>                operation size implementation median_seconds  itr_per_sec
#> 1        Matrix multiply  500         base R   0.0024753340 2.619792e+02
#> 2        Matrix multiply  500            mlx   0.0011954370 6.247334e+02
#> 3        Matrix multiply 1000         base R   0.0190236720 4.395051e+01
#> 4        Matrix multiply 1000            mlx   0.0046640985 1.679161e+02
#> 5        Matrix multiply 2000         base R   0.1461716420 6.585680e+00
#> 6        Matrix multiply 2000            mlx   0.0177136195 4.317815e+01
#> 7        Matrix addition  500         base R   0.0002410800 1.163390e+03
#> 8        Matrix addition  500            mlx   0.0006984350 1.166048e+03
#> 9        Matrix addition 1000         base R   0.0011685820 4.514375e+02
#> 10       Matrix addition 1000            mlx   0.0018526260 4.163320e+02
#> 11       Matrix addition 2000         base R   0.0030980215 2.468754e+02
#> 12       Matrix addition 2000            mlx   0.0084501615 1.177255e+02
#> 13  Subset rows (vector)  500         base R   0.0003724645 1.455555e+03
#> 14  Subset rows (vector)  500            mlx   0.0007106940 1.319652e+03
#> 15  Subset rows (vector) 1000         base R   0.0014299980 5.461792e+02
#> 16  Subset rows (vector) 1000            mlx   0.0015627970 6.208401e+02
#> 17  Subset rows (vector) 2000         base R   0.0064935185 1.336738e+02
#> 18  Subset rows (vector) 2000            mlx   0.0044347445 1.980558e+02
#> 19 Subset (matrix index)  500         base R   0.0000063550 1.305289e+05
#> 20 Subset (matrix index)  500            mlx   0.0008210660 1.108016e+03
#> 21 Subset (matrix index) 1000         base R   0.0000092660 8.186951e+04
#> 22 Subset (matrix index) 1000            mlx   0.0010499895 9.337375e+02
#> 23 Subset (matrix index) 2000         base R   0.0000173020 5.146663e+04
#> 24 Subset (matrix index) 2000            mlx   0.0031220270 2.404876e+02
#> 25                   Sum  500         base R   0.0002313220 4.210233e+03
#> 26                   Sum  500            mlx   0.0004579495 2.116990e+03
#> 27                   Sum 1000         base R   0.0009290190 1.062690e+03
#> 28                   Sum 1000            mlx   0.0005387810 1.765635e+03
#> 29                   Sum 2000         base R   0.0040279425 2.479852e+02
#> 30                   Sum 2000            mlx   0.0008035180 1.229896e+03
#> 31                  Mean  500         base R   0.0004882895 2.004472e+03
#> 32                  Mean  500            mlx   0.0004866700 2.022726e+03
#> 33                  Mean 1000         base R   0.0019941170 4.940310e+02
#> 34                  Mean 1000            mlx   0.0005352960 1.839222e+03
#> 35                  Mean 2000         base R   0.0078746855 1.257134e+02
#> 36                  Mean 2000            mlx   0.0007993770 1.231332e+03
#> 37              Row sums  500         base R   0.0000453460 2.072120e+04
#> 38              Row sums  500            mlx   0.0004914465 2.007851e+03
#> 39              Row sums 1000         base R   0.0001723640 5.457683e+03
#> 40              Row sums 1000            mlx   0.0005407900 1.821476e+03
#> 41              Row sums 2000         base R   0.0007112885 1.310104e+03
#> 42              Row sums 2000            mlx   0.0007921610 1.233006e+03
#> 43             Row means  500         base R   0.0000460430 2.060420e+04
#> 44             Row means  500            mlx   0.0004891915 2.022602e+03
#> 45             Row means 1000         base R   0.0001755210 5.597488e+03
#> 46             Row means 1000            mlx   0.0005519420 1.786078e+03
#> 47             Row means 2000         base R   0.0006953600 1.424234e+03
#> 48             Row means 2000            mlx   0.0008190365 1.199570e+03
#> 49            tcrossprod  500         base R   0.0014920925 5.413073e+02
#> 50            tcrossprod  500            mlx   0.0009262720 8.815322e+02
#> 51            tcrossprod 1000         base R   0.0105721780 8.289675e+01
#> 52            tcrossprod 1000            mlx   0.0030424460 3.226751e+02
#> 53            tcrossprod 2000         base R   0.0725093815 1.366682e+01
#> 54            tcrossprod 2000            mlx   0.0121604975 7.875306e+01
#> 55               scale()  500         base R   0.0060329040 1.204775e+02
#> 56               scale()  500            mlx   0.0013830940 7.713564e+02
#> 57               scale() 1000         base R   0.0273437200 2.532245e+01
#> 58               scale() 1000            mlx   0.0028047485 3.236396e+02
#> 59               scale() 2000         base R   0.1832500740 5.273578e+00
#> 60               scale() 2000            mlx   0.0207369800 4.777054e+01
#> 61          Solve Ax = b  500         base R   0.0027026380 3.625851e+02
#> 62          Solve Ax = b  500            mlx   0.0042304005 2.442463e+02
#> 63          Solve Ax = b 1000         base R   0.0130645270 5.596831e+01
#> 64          Solve Ax = b 1000            mlx   0.0243911665 3.975141e+01
#> 65          Solve Ax = b 2000         base R   0.0701741650 1.401404e+01
#> 66          Solve Ax = b 2000            mlx   0.1777443890 5.322684e+00
#> 67             Backsolve  500         base R   0.0000477650 2.094512e+04
#> 68             Backsolve  500            mlx   0.0013753040 7.348256e+02
#> 69             Backsolve 1000         base R   0.0001507570 6.461902e+03
#> 70             Backsolve 1000            mlx   0.0091550950 1.074366e+02
#> 71             Backsolve 2000         base R   0.0009305360 1.048274e+03
#> 72             Backsolve 2000            mlx   0.0703910345 1.410861e+01
#> 73              Cholesky  500         base R   0.0014434665 6.445818e+02
#> 74              Cholesky  500            mlx   0.0007095050 1.258827e+03
#> 75              Cholesky 1000         base R   0.0081364090 1.066146e+02
#> 76              Cholesky 1000            mlx   0.0043486650 2.272458e+02
#> 77              Cholesky 2000         base R   0.0678114170 1.492333e+01
#> 78              Cholesky 2000            mlx   0.0212427970 4.272313e+01
#> 79              chol2inv  500         base R   0.0031944740 3.055230e+02
#> 80              chol2inv  500            mlx   0.0020928860 4.734437e+02
#> 81              chol2inv 1000         base R   0.0229165400 4.246879e+01
#> 82              chol2inv 1000            mlx   0.0150880000 6.637836e+01
#> 83              chol2inv 2000         base R   0.1685920000 5.853495e+00
#> 84              chol2inv 2000            mlx   0.1129528680 8.734245e+00
#> 85     SVD (values only)  500         base R   0.0324149075 3.088924e+01
#> 86     SVD (values only)  500            mlx   0.0195502350 4.937810e+01
#> 87     SVD (values only) 1000         base R   0.1661634060 6.018172e+00
#> 88     SVD (values only) 1000            mlx   0.0960484860 1.058381e+01
#> 89     SVD (values only) 2000         base R   1.3880643480 7.204277e-01
#> 90     SVD (values only) 2000            mlx   0.6391082870 1.564680e+00
#> 91              Diagonal  500         base R   0.0000066420 7.217236e+04
#> 92              Diagonal  500            mlx   0.0000523160 1.507311e+04
#> 93              Diagonal 1000         base R   0.0000111930 5.304630e+04
#> 94              Diagonal 1000            mlx   0.0000651900 1.495894e+04
#> 95              Diagonal 2000         base R   0.0000166050 3.527051e+04
#> 96              Diagonal 2000            mlx   0.0000839680 1.143221e+04
#>    mem_alloc_bytes
#> 1          2000048
#> 2            63968
#> 3          8000048
#> 4                0
#> 5         32000048
#> 6                0
#> 7          2000048
#> 8            57224
#> 9          8000048
#> 10               0
#> 11        32000048
#> 12               0
#> 13         2002096
#> 14          125024
#> 15         8004096
#> 16           36384
#> 17        32008096
#> 18           72384
#> 19            6096
#> 20          190752
#> 21           12096
#> 22          124912
#> 23           24096
#> 24          248912
#> 25               0
#> 26           53472
#> 27               0
#> 28               0
#> 29               0
#> 30               0
#> 31               0
#> 32            2656
#> 33               0
#> 34               0
#> 35               0
#> 36               0
#> 37            8152
#> 38           18096
#> 39            8048
#> 40               0
#> 41           16048
#> 42               0
#> 43            8152
#> 44            3368
#> 45            8048
#> 46               0
#> 47           16048
#> 48               0
#> 49         2000048
#> 50           11952
#> 51         8000048
#> 52               0
#> 53        32000048
#> 54               0
#> 55        21277632
#> 56           78840
#> 57        84424480
#> 58               0
#> 59       336848480
#> 60               0
#> 61         2039440
#> 62           18016
#> 63         8044216
#> 64               0
#> 65        32088216
#> 66               0
#> 67           10136
#> 68           30840
#> 69            8048
#> 70               0
#> 71           16048
#> 72               0
#> 73         2000048
#> 74           10776
#> 75         8000048
#> 76               0
#> 77        32000048
#> 78               0
#> 79         2003712
#> 80           12272
#> 81         8000048
#> 82               0
#> 83        32000048
#> 84               0
#> 85         6332952
#> 86           31680
#> 87        24576408
#> 88               0
#> 89        97152408
#> 90               0
#> 91           12192
#> 92           13768
#> 93           24192
#> 94               0
#> 95           48192
#> 96               0
```
