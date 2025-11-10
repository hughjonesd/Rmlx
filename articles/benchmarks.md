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
#> 1        Matrix multiply  500         base R   0.0056192550 1.084004e+02
#> 2        Matrix multiply  500            mlx   0.0025665385 3.141945e+02
#> 3        Matrix multiply 1000         base R   0.0354485590 2.703727e+01
#> 4        Matrix multiply 1000            mlx   0.0066531725 1.422385e+02
#> 5        Matrix multiply 2000         base R   0.2342546480 4.423731e+00
#> 6        Matrix multiply 2000            mlx   0.0299831975 3.167681e+01
#> 7        Matrix addition  500         base R   0.0009590720 3.837858e+02
#> 8        Matrix addition  500            mlx   0.0019914110 4.387120e+02
#> 9        Matrix addition 1000         base R   0.0022187560 1.435514e+02
#> 10       Matrix addition 1000            mlx   0.0036578560 1.942551e+02
#> 11       Matrix addition 2000         base R   0.0074757760 8.294704e+01
#> 12       Matrix addition 2000            mlx   0.0094056870 5.454110e+01
#> 13  Subset rows (vector)  500         base R   0.0005074160 8.853940e+02
#> 14  Subset rows (vector)  500            mlx   0.0015549250 5.152971e+02
#> 15  Subset rows (vector) 1000         base R   0.0027709030 1.902972e+02
#> 16  Subset rows (vector) 1000            mlx   0.0026064520 2.869706e+02
#> 17  Subset rows (vector) 2000         base R   0.0105857900 6.954650e+01
#> 18  Subset rows (vector) 2000            mlx   0.0069742230 8.215924e+01
#> 19 Subset (matrix index)  500         base R   0.0000058630 7.491876e+04
#> 20 Subset (matrix index)  500            mlx   0.0018726340 4.111944e+02
#> 21 Subset (matrix index) 1000         base R   0.0000131200 3.218164e+04
#> 22 Subset (matrix index) 1000            mlx   0.0039641055 2.462974e+02
#> 23 Subset (matrix index) 2000         base R   0.0000199260 2.688333e+04
#> 24 Subset (matrix index) 2000            mlx   0.0078175930 1.082103e+02
#> 25                   Sum  500         base R   0.0002593660 3.095553e+03
#> 26                   Sum  500            mlx   0.0013826020 6.250225e+02
#> 27                   Sum 1000         base R   0.0010883040 7.533123e+02
#> 28                   Sum 1000            mlx   0.0021074205 3.827191e+02
#> 29                   Sum 2000         base R   0.0053285240 1.835353e+02
#> 30                   Sum 2000            mlx   0.0018638395 4.702408e+02
#> 31                  Mean  500         base R   0.0005322210 1.611984e+03
#> 32                  Mean  500            mlx   0.0015373155 5.550495e+02
#> 33                  Mean 1000         base R   0.0024506930 3.870044e+02
#> 34                  Mean 1000            mlx   0.0015654210 5.894695e+02
#> 35                  Mean 2000         base R   0.0094396350 1.014327e+02
#> 36                  Mean 2000            mlx   0.0008948250 8.474970e+02
#> 37              Row sums  500         base R   0.0000483390 1.323692e+04
#> 38              Row sums  500            mlx   0.0011219035 7.599207e+02
#> 39              Row sums 1000         base R   0.0001903425 4.269873e+03
#> 40              Row sums 1000            mlx   0.0010844090 7.879734e+02
#> 41              Row sums 2000         base R   0.0008138500 1.132003e+03
#> 42              Row sums 2000            mlx   0.0015031215 5.436795e+02
#> 43             Row means  500         base R   0.0000462480 1.937720e+04
#> 44             Row means  500            mlx   0.0007639735 1.103933e+03
#> 45             Row means 1000         base R   0.0002046925 4.222343e+03
#> 46             Row means 1000            mlx   0.0010396370 7.559658e+02
#> 47             Row means 2000         base R   0.0008483105 1.023459e+03
#> 48             Row means 2000            mlx   0.0021692280 4.478188e+02
#> 49            tcrossprod  500         base R   0.0030284650 2.618998e+02
#> 50            tcrossprod  500            mlx   0.0018302195 5.024318e+02
#> 51            tcrossprod 1000         base R   0.0180201970 5.120097e+01
#> 52            tcrossprod 1000            mlx   0.0055295470 1.638367e+02
#> 53            tcrossprod 2000         base R   0.1257296570 8.144274e+00
#> 54            tcrossprod 2000            mlx   0.0219204245 4.384123e+01
#> 55               scale()  500         base R   0.0106382085 7.442895e+01
#> 56               scale()  500            mlx   0.0023493820 3.610512e+02
#> 57               scale() 1000         base R   0.0437081525 1.968251e+01
#> 58               scale() 1000            mlx   0.0095465835 9.535593e+01
#> 59               scale() 2000         base R   0.2149948980 4.187286e+00
#> 60               scale() 2000            mlx   0.0230035420 5.081866e+01
#> 61          Solve Ax = b  500         base R   0.0059029135 1.514169e+02
#> 62          Solve Ax = b  500            mlx   0.0078360430 9.138512e+01
#> 63          Solve Ax = b 1000         base R   0.0309259310 2.815386e+01
#> 64          Solve Ax = b 1000            mlx   0.0539104080 1.789232e+01
#> 65          Solve Ax = b 2000         base R   0.2229500460 4.831977e+00
#> 66          Solve Ax = b 2000            mlx   0.3373213090 2.990315e+00
#> 67             Backsolve  500         base R   0.0000542430 1.298675e+04
#> 68             Backsolve  500            mlx   0.0019218750 4.585348e+02
#> 69             Backsolve 1000         base R   0.0001895020 4.047490e+03
#> 70             Backsolve 1000            mlx   0.0178721460 5.161504e+01
#> 71             Backsolve 2000         base R   0.0020281470 4.250409e+02
#> 72             Backsolve 2000            mlx   0.1108600230 9.154800e+00
#> 73              Cholesky  500         base R   0.0016547600 4.828347e+02
#> 74              Cholesky  500            mlx   0.0009377520 8.436253e+02
#> 75              Cholesky 1000         base R   0.0101729610 7.304026e+01
#> 76              Cholesky 1000            mlx   0.0052290990 1.679068e+02
#> 77              Cholesky 2000         base R   0.0817309580 1.153969e+01
#> 78              Cholesky 2000            mlx   0.0232665570 4.203006e+01
#> 79              chol2inv  500         base R   0.0031999270 2.995048e+02
#> 80              chol2inv  500            mlx   0.0028060810 3.290458e+02
#> 81              chol2inv 1000         base R   0.0238582280 3.362282e+01
#> 82              chol2inv 1000            mlx   0.0202708510 5.003663e+01
#> 83              chol2inv 2000         base R   0.2794457090 3.350494e+00
#> 84              chol2inv 2000            mlx   0.1535465170 6.666329e+00
#> 85     SVD (values only)  500         base R   0.0356691390 2.683829e+01
#> 86     SVD (values only)  500            mlx   0.0220722270 4.272537e+01
#> 87     SVD (values only) 1000         base R   0.2887290520 3.463455e+00
#> 88     SVD (values only) 1000            mlx   0.2036088495 4.911378e+00
#> 89     SVD (values only) 2000         base R   3.1076243560 3.217892e-01
#> 90     SVD (values only) 2000            mlx   1.5034447850 6.651392e-01
#> 91              Diagonal  500         base R   0.0000152930 4.284010e+04
#> 92              Diagonal  500            mlx   0.0000898310 6.877440e+03
#> 93              Diagonal 1000         base R   0.0000115210 4.754767e+04
#> 94              Diagonal 1000            mlx   0.0000796425 7.751364e+03
#> 95              Diagonal 2000         base R   0.0000186550 2.741781e+04
#> 96              Diagonal 2000            mlx   0.0000966370 7.531201e+03
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
