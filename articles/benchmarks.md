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
#> 1        Matrix multiply  500         base R   0.0050193840 1.742726e+02
#> 2        Matrix multiply  500            mlx   0.0023854620 3.846013e+02
#> 3        Matrix multiply 1000         base R   0.0349343370 2.830907e+01
#> 4        Matrix multiply 1000            mlx   0.0063950160 1.459129e+02
#> 5        Matrix multiply 2000         base R   0.2927612380 3.610433e+00
#> 6        Matrix multiply 2000            mlx   0.0240113425 3.863392e+01
#> 7        Matrix addition  500         base R   0.0005588710 3.638532e+02
#> 8        Matrix addition  500            mlx   0.0017171415 4.685872e+02
#> 9        Matrix addition 1000         base R   0.0029643000 1.932511e+02
#> 10       Matrix addition 1000            mlx   0.0044482540 2.103953e+02
#> 11       Matrix addition 2000         base R   0.0069194265 1.093501e+02
#> 12       Matrix addition 2000            mlx   0.0115670840 7.455226e+01
#> 13  Subset rows (vector)  500         base R   0.0005361570 8.668606e+02
#> 14  Subset rows (vector)  500            mlx   0.0019079350 4.646896e+02
#> 15  Subset rows (vector) 1000         base R   0.0018945485 4.137142e+02
#> 16  Subset rows (vector) 1000            mlx   0.0016058060 4.260506e+02
#> 17  Subset rows (vector) 2000         base R   0.0108336965 8.696214e+01
#> 18  Subset rows (vector) 2000            mlx   0.0079647010 8.923212e+01
#> 19 Subset (matrix index)  500         base R   0.0000058630 8.228705e+04
#> 20 Subset (matrix index)  500            mlx   0.0017674690 5.437482e+02
#> 21 Subset (matrix index) 1000         base R   0.0000105370 4.786412e+04
#> 22 Subset (matrix index) 1000            mlx   0.0032195250 2.878275e+02
#> 23 Subset (matrix index) 2000         base R   0.0000189830 3.973586e+04
#> 24 Subset (matrix index) 2000            mlx   0.0050784445 1.396728e+02
#> 25                   Sum  500         base R   0.0002524370 3.794500e+03
#> 26                   Sum  500            mlx   0.0009230945 9.493091e+02
#> 27                   Sum 1000         base R   0.0010172510 9.831240e+02
#> 28                   Sum 1000            mlx   0.0008582940 1.047312e+03
#> 29                   Sum 2000         base R   0.0041417585 2.066437e+02
#> 30                   Sum 2000            mlx   0.0007783030 1.266304e+03
#> 31                  Mean  500         base R   0.0004681380 2.074708e+03
#> 32                  Mean  500            mlx   0.0004456700 2.108055e+03
#> 33                  Mean 1000         base R   0.0019398330 5.105870e+02
#> 34                  Mean 1000            mlx   0.0006018800 1.414074e+03
#> 35                  Mean 2000         base R   0.0075115690 1.314297e+02
#> 36                  Mean 2000            mlx   0.0007894550 1.240865e+03
#> 37              Row sums  500         base R   0.0000465760 1.772464e+04
#> 38              Row sums  500            mlx   0.0006047910 1.542552e+03
#> 39              Row sums 1000         base R   0.0001852790 5.284556e+03
#> 40              Row sums 1000            mlx   0.0005605520 1.720991e+03
#> 41              Row sums 2000         base R   0.0007950720 1.171308e+03
#> 42              Row sums 2000            mlx   0.0010007690 9.225728e+02
#> 43             Row means  500         base R   0.0000466170 1.952324e+04
#> 44             Row means  500            mlx   0.0005866690 1.659640e+03
#> 45             Row means 1000         base R   0.0001829830 5.333633e+03
#> 46             Row means 1000            mlx   0.0005593220 1.769197e+03
#> 47             Row means 2000         base R   0.0007678070 1.288549e+03
#> 48             Row means 2000            mlx   0.0009747750 9.709249e+02
#> 49            tcrossprod  500         base R   0.0025003850 3.528191e+02
#> 50            tcrossprod  500            mlx   0.0017764480 5.576358e+02
#> 51            tcrossprod 1000         base R   0.0180093115 5.599148e+01
#> 52            tcrossprod 1000            mlx   0.0036438545 2.579169e+02
#> 53            tcrossprod 2000         base R   0.1000121200 1.012811e+01
#> 54            tcrossprod 2000            mlx   0.0185162970 5.404302e+01
#> 55               scale()  500         base R   0.0063189200 1.302327e+02
#> 56               scale()  500            mlx   0.0022041190 3.910823e+02
#> 57               scale() 1000         base R   0.0525354730 1.768157e+01
#> 58               scale() 1000            mlx   0.0056885860 1.708040e+02
#> 59               scale() 2000         base R   0.1482356640 7.081711e+00
#> 60               scale() 2000            mlx   0.0115465635 6.930481e+01
#> 61          Solve Ax = b  500         base R   0.0048934320 1.675599e+02
#> 62          Solve Ax = b  500            mlx   0.0061884580 1.412916e+02
#> 63          Solve Ax = b 1000         base R   0.0174851880 4.719789e+01
#> 64          Solve Ax = b 1000            mlx   0.0259267395 3.830638e+01
#> 65          Solve Ax = b 2000         base R   0.1158095020 8.399765e+00
#> 66          Solve Ax = b 2000            mlx   0.2272330290 4.436153e+00
#> 67             Backsolve  500         base R   0.0000437880 2.183377e+04
#> 68             Backsolve  500            mlx   0.0011923210 7.642856e+02
#> 69             Backsolve 1000         base R   0.0001989730 4.806241e+03
#> 70             Backsolve 1000            mlx   0.0106372655 9.337964e+01
#> 71             Backsolve 2000         base R   0.0012561990 8.519892e+02
#> 72             Backsolve 2000            mlx   0.0813118970 1.184090e+01
#> 73              Cholesky  500         base R   0.0023989100 3.553342e+02
#> 74              Cholesky  500            mlx   0.0012289955 8.257075e+02
#> 75              Cholesky 1000         base R   0.0095132915 1.013678e+02
#> 76              Cholesky 1000            mlx   0.0026129300 3.055058e+02
#> 77              Cholesky 2000         base R   0.0633836630 1.487158e+01
#> 78              Cholesky 2000            mlx   0.0197353090 4.696218e+01
#> 79              chol2inv  500         base R   0.0030582720 3.088156e+02
#> 80              chol2inv  500            mlx   0.0018265910 5.336827e+02
#> 81              chol2inv 1000         base R   0.0204288650 4.755962e+01
#> 82              chol2inv 1000            mlx   0.0124003680 8.020957e+01
#> 83              chol2inv 2000         base R   0.1735785020 5.837254e+00
#> 84              chol2inv 2000            mlx   0.1028988890 9.631307e+00
#> 85     SVD (values only)  500         base R   0.0254169250 3.887208e+01
#> 86     SVD (values only)  500            mlx   0.0167963880 5.927340e+01
#> 87     SVD (values only) 1000         base R   0.2396846265 4.172149e+00
#> 88     SVD (values only) 1000            mlx   0.1179425270 8.229783e+00
#> 89     SVD (values only) 2000         base R   1.6843731110 5.936927e-01
#> 90     SVD (values only) 2000            mlx   0.7042644210 1.419921e+00
#> 91              Diagonal  500         base R   0.0000104140 4.238395e+04
#> 92              Diagonal  500            mlx   0.0000597780 1.493437e+04
#> 93              Diagonal 1000         base R   0.0000102500 5.553831e+04
#> 94              Diagonal 1000            mlx   0.0000701920 1.295298e+04
#> 95              Diagonal 2000         base R   0.0000161540 3.855850e+04
#> 96              Diagonal 2000            mlx   0.0000990970 9.207110e+03
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
