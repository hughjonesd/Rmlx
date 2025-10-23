
# Rmlx

<!-- badges: start -->

[![Codecov test
coverage](https://codecov.io/gh/hughjonesd/Rmlx/graph/badge.svg)](https://app.codecov.io/gh/hughjonesd/Rmlx)
<!-- badges: end -->

R interface to Apple’s MLX (Machine Learning eXchange) library for
GPU-accelerated array operations on Apple Silicon.

## Overview

Rmlx provides an R interface to Apple’s MLX framework, enabling
high-performance GPU computing on Apple Silicon (M1, M2, M3+) using the
Metal backend. The package implements lazy evaluation and familiar R
syntax through S3 method dispatch.

**Status**: Phase 1 implementation complete (arrays, operations,
evaluation, tests, documentation). Phase 2 (autodiff, optimizers) not
yet implemented.

## Requirements

- macOS on Apple Silicon (M1/M2/M3 or later)
- MLX C/C++ library installed
- R \>= 4.1.0
- Rcpp \>= 1.0.10

## Installation

### Install MLX

First, install the MLX library:

``` bash
# Option 1: Homebrew (if available)
brew install mlx

# Option 2: Build from source
git clone https://github.com/ml-explore/mlx.git
cd mlx
mkdir build && cd build
cmake ..
make
sudo make install
```

### Install Rmlx

``` r
# Install from source
devtools::install()

# Or with custom MLX paths
Sys.setenv(MLX_INCLUDE = "/path/to/mlx/include")
Sys.setenv(MLX_LIB_DIR = "/path/to/mlx/lib")
devtools::install()
```

## Features

### Lazy Evaluation

Operations are recorded but not executed until explicitly evaluated:

``` r
library(Rmlx)
#> 
#> Attaching package: 'Rmlx'
#> The following object is masked from 'package:stats':
#> 
#>     fft
#> The following objects are masked from 'package:base':
#> 
#>     colMeans, colSums, rowMeans, rowSums

x <- as_mlx(matrix(1:100, 10, 10))
y <- as_mlx(matrix(101:200, 10, 10))

# Lazy - not computed yet
z <- x + y * 2

# Force evaluation
mlx_eval(z)

# Or convert to R (automatically evaluates)
result <- as.matrix(z)

# Wait for queued GPU work (useful when timing)
mlx_synchronize("gpu")

# Simple aggregate checks
sum(z)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 35150
mean(z)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 351.5
```

### Arithmetic Operations

Standard R operators work seamlessly:

``` r
x <- as_mlx(matrix(1:12, 3, 4))
y <- as_mlx(matrix(13:24, 3, 4))

# Element-wise operations
sum_xy <- x + y
diff_xy <- x - y
prod_xy <- x * y
quot_xy <- x / y
pow_xy <- x ^ 2

# Comparisons
lt <- x < y
eq <- x == y

# Bring results back to R
as.matrix(sum_xy)
#>      [,1] [,2] [,3] [,4]
#> [1,]   14   20   26   32
#> [2,]   16   22   28   34
#> [3,]   18   24   30   36
as.matrix(lt)
#>      [,1] [,2] [,3] [,4]
#> [1,] TRUE TRUE TRUE TRUE
#> [2,] TRUE TRUE TRUE TRUE
#> [3,] TRUE TRUE TRUE TRUE
```

### Matrix Operations

``` r
a <- as_mlx(matrix(1:6, 2, 3))
b <- as_mlx(matrix(1:6, 3, 2))

# Matrix multiplication
c <- a %*% b
as.matrix(c)
#>      [,1] [,2]
#> [1,]   22   49
#> [2,]   28   64
```

Advanced decompositions mirror base R:

### Random Sampling

``` r
random_tensor <- mlx_rand_uniform(c(512, 512), min = -1, max = 1)
random_tensor
#> mlx array [512 x 512]
#>   dtype: float32
#>   device: gpu
#>   (262144 elements, not shown)
```

### Data Transformations

Common ranking helpers are available under the `mlx_*` prefix; note that
MLX indices are zero-based.

``` r
scores <- as_mlx(c(0.1, 0.7, 0.4, 0.9))
as.matrix(mlx_sort(scores))
#> [1] 0.1 0.4 0.7 0.9
as.matrix(mlx_topk(scores, 2))
#> [1] 0.7 0.9
as.matrix(mlx_argmax(scores))
#> [1] 3
```

``` r
qr_res <- qr(a)
svd_res <- svd(a)
chol_res <- chol(as_mlx(crossprod(matrix(1:6, 3, 2))))
fft_res <- fft(a)

# Inspect outputs
qr_res$Q
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]       [,2]
#> [1,] -0.4472135 -0.8944272
#> [2,] -0.8944272  0.4472136
svd_res$d
#> [1] 9.5255181 0.5143006
as.matrix(chol_res)
#>          [,1]     [,2]
#> [1,] 3.741657 8.552360
#> [2,] 0.000000 1.963962
```

### Differentiation

``` r
loss <- function(w, x, y) {
  preds <- x %*% w
  resids <- preds - y
  sum(resids * resids) / length(y)
}

x <- as_mlx(matrix(rnorm(20), 5, 4))
y <- as_mlx(matrix(rnorm(5), 5, 1))
w <- as_mlx(matrix(0, 4, 1))

grads <- mlx_grad(loss, w, x, y)

# Inspect gradient
as.matrix(grads[[1]])
#>             [,1]
#> [1,] 0.014563746
#> [2,] 1.036490083
#> [3,] 1.091995597
#> [4,] 0.007043276

# Simple SGD loop
model <- mlx_linear(4, 1, bias = FALSE)
opt <- mlx_optimizer_sgd(mlx_parameters(model), lr = 0.1)
loss_fn <- function(mod, data_x, data_y) {
  preds <- mlx_forward(mod, data_x)
  resids <- preds - data_y
  sum(resids * resids) / length(data_y)
}
for (step in 1:50) {
  mlx_train_step(model, loss_fn, opt, x, y)
}

# Check final loss
final_loss <- mlx_forward(model, x)
mean((final_loss - y) * (final_loss - y))
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 0.3989661
```

### Reductions

``` r
x <- as_mlx(matrix(1:100, 10, 10))

# Overall reductions
sum(x)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 5050
mean(x)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 50.5

# Column/row means
colMeans(x)
#> mlx array [10]
#>   dtype: float32
#>   device: gpu
#>   values:
#>  [1]  5.5 15.5 25.5 35.5 45.5 55.5 65.5 75.5 85.5 95.5
rowMeans(x)
#> mlx array [10]
#>   dtype: float32
#>   device: gpu
#>   values:
#>  [1] 46 47 48 49 50 51 52 53 54 55

# Cumulative operations flatten column-major
as.vector(cumsum(x))
#>   [1]    1    3    6   10   15   21   28   36   45   55   66   78   91  105  120
#>  [16]  136  153  171  190  210  231  253  276  300  325  351  378  406  435  465
#>  [31]  496  528  561  595  630  666  703  741  780  820  861  903  946  990 1035
#>  [46] 1081 1128 1176 1225 1275 1326 1378 1431 1485 1540 1596 1653 1711 1770 1830
#>  [61] 1891 1953 2016 2080 2145 2211 2278 2346 2415 2485 2556 2628 2701 2775 2850
#>  [76] 2926 3003 3081 3160 3240 3321 3403 3486 3570 3655 3741 3828 3916 4005 4095
#>  [91] 4186 4278 4371 4465 4560 4656 4753 4851 4950 5050
```

### Indexing

``` r
x <- as_mlx(matrix(1:100, 10, 10))

# Subset
x[1:5, 1:5]
#> mlx array [5 x 5]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    1   11   21   31   41
#> [2,]    2   12   22   32   42
#> [3,]    3   13   23   33   43
#> [4,]    4   14   24   34   44
#> [5,]    5   15   25   35   45
x[1, ]
#> mlx array [1 x 10]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#> [1,]    1   11   21   31   41   51   61   71   81    91
x[, 1]
#> mlx array [10 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>       [,1]
#>  [1,]    1
#>  [2,]    2
#>  [3,]    3
#>  [4,]    4
#>  [5,]    5
#>  [6,]    6
#>  [7,]    7
#>  [8,]    8
#>  [9,]    9
#> [10,]   10
```

### Device Management

``` r
# Check/set default device
mlx_default_device()           # "gpu"
#> [1] "gpu"
mlx_default_device("cpu")      # Switch to CPU
#> [1] "cpu"
mlx_default_device("gpu")      # Back to GPU
#> [1] "gpu"

# Create on specific device
x_gpu <- as_mlx(matrix(1:12, 3, 4), device = "gpu")
x_cpu <- as_mlx(matrix(1:12, 3, 4), device = "cpu")
```

> **Precision note:** Numeric inputs are stored in `float32`. Requests
> for `dtype = "float64"` are downcast with a warning. Logical inputs
> are stored as MLX `bool` tensors (logical `NA` values are not
> supported). Complex inputs are stored as `complex64` (single-precision
> real/imaginary parts). Use base R arrays if you require double
> precision arithmetic.

## Data Types

Supported dtype:

- `float32` for numeric data (default)
- `bool` for logical data

``` r
x_f32 <- as_mlx(matrix(1:12, 3, 4), dtype = "float32")
logical_mat <- as_mlx(matrix(c(TRUE, FALSE, TRUE, TRUE), 2, 2))
```

## Documentation

- Package documentation: `?Rmlx`
- Getting started vignette:
  `vignette("getting-started", package = "Rmlx")`
- Function help: `?as_mlx`, `?mlx_eval`, etc.

## Testing

Tests use testthat and compare against base R results:

``` r
# Run all tests
devtools::test()

# Run specific test file
devtools::test_file("tests/testthat/test-ops.R")
```

Tests skip gracefully if MLX is not available or the package fails to
load.
