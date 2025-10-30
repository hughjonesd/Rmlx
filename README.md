
# Rmlx

<!-- badges: start -->

[![Codecov test
coverage](https://codecov.io/gh/hughjonesd/Rmlx/graph/badge.svg)](https://app.codecov.io/gh/hughjonesd/Rmlx)
<!-- badges: end -->

R interface to Apple’s MLX (Machine Learning eXchange) library.

## Overview

Rmlx provides an R interface to Apple’s [MLX
framework](https://ml-explore.github.io/mlx/), enabling high-performance
GPU computing on Apple Silicon.

This package was vibe-coded with Claude/OpenAI Codex in a week. Use at
your own risk! Much of the C++ API has been implemented, but not
python-only features such as large neural network layers.

## Motivation

Modern Macs have a GPU, which is great for performing matrix operations.
Statistics uses a lot of matrix operations. But until now, there has
been no way for R on the Mac to use the GPU. Rmlx changes that.

## Requirements

- macOS on Apple Silicon *or* Linux with CUDA *or* MacOS/Linux for a
  CPU-only build

## Installation

`brew install mlx`

or the Linux equivalent. Then just install the package as normal:

``` r
# install.packages("remotes")
remotes::install("hughjonesd/Rmlx")
```

Alternatively, you can build mlx from the
[source](https://github.com/ml-explore/mlx).

## Features

### Fast GPU Operations

``` r

library(Rmlx)
#> 
#> Attaching package: 'Rmlx'
#> The following object is masked from 'package:stats':
#> 
#>     fft
#> The following objects are masked from 'package:base':
#> 
#>     chol2inv, colMeans, colSums, diag, outer, rowMeans, rowSums, svd

A <- matrix(rnorm(1e6), 1e3, 1e3)
system.time(solve(A))
#>    user  system elapsed 
#>   0.395   0.002   0.399
system.time(solve(as_mlx(A)))
#>    user  system elapsed 
#>   0.048   0.055   0.104
```

### Lazy Evaluation

Operations are recorded but not executed until explicitly evaluated:

``` r

x <- as_mlx(matrix(1:25, 5, 5))
y <- as_mlx(matrix(101:125, 5, 5))

# Lazy - not computed yet
z <- x + y * 2

# Force evaluation
mlx_eval(z)

# Or convert to R (automatically evaluates)
as.matrix(z)
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]  203  218  233  248  263
#> [2,]  206  221  236  251  266
#> [3,]  209  224  239  254  269
#> [4,]  212  227  242  257  272
#> [5,]  215  230  245  260  275
```

### Device Management

M series chips have shared memory between the CPU and GPU, so switching
between devices is costless.

``` r
# Check/set default device
dev <- mlx_default_device()           
mlx_default_device("cpu")    # Switch to CPU
mlx_default_device(dev)      # Back to GPU

# Create on specific device
x_gpu <- as_mlx(matrix(1:12, 3, 4), device = "gpu")
x_cpu <- as_mlx(matrix(1:12, 3, 4), device = "cpu")
```

### Subsetting

Subsetting works like base R:

``` r
x <- as_mlx(matrix(1:9, 3, 3))
x[1:2, 1:2]
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    1    4
#> [2,]    2    5

# drop = FALSE by default
x[1, ]
#> mlx array [1 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    1    4    7

logical_mask <- c(TRUE, FALSE, TRUE)
x[logical_mask, ]
#> mlx array [2 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    1    4    7
#> [2,]    3    6    9

# subset assignment

x[, 2] <- c(0, 0, 0)
x
#> mlx array [3 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    1    0    7
#> [2,]    2    0    8
#> [3,]    3    0    9
```

### Arithmetic

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

Many base R matrix functions have mlx-specific methods:

``` r
a <- as_mlx(matrix(1:6, 2, 3))
b <- as_mlx(matrix(1:6, 3, 2))

# rbind, cbind, transpose
rbind(a, t(b))
#> mlx array [4 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    1    3    5
#> [2,]    2    4    6
#> [3,]    1    2    3
#> [4,]    4    5    6
cbind(a, t(b))
#> mlx array [2 x 6]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4] [,5] [,6]
#> [1,]    1    3    5    1    2    3
#> [2,]    2    4    6    4    5    6

# Matrix multiplication
a %*% b
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]   22   49
#> [2,]   28   64

# Reductions
sum(a)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 21
mean(a)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 3.5
colMeans(a)
#> mlx array [3]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1.5 3.5 5.5
rowMeans(a)
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 3 4

# Cumulative operations flatten column-major
as.vector(cumsum(a))
#> [1]  1  3  6 10 15 21

qr_res <- qr(a)
svd_res <- svd(a)
chol_res <- chol(as_mlx(crossprod(matrix(1:6, 3, 2))))
fft_res <- fft(a)

qr_res$Q
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]       [,2]
#> [1,] -0.4472135 -0.8944272
#> [2,] -0.8944272  0.4472136
svd_res$d
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 9.5255194 0.5143015
chol_res
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>          [,1]     [,2]
#> [1,] 3.741657 8.552360
#> [2,] 0.000000 1.963962
```

### Random Sampling

``` r
mlx_rand_uniform(c(3, 3), min = -1, max = 1)
#> mlx array [3 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]       [,2]       [,3]
#> [1,] -0.88357937 -0.3713405 -0.0342840
#> [2,]  0.09261358 -0.2968941 -0.4256831
#> [3,]  0.72074342  0.5229734 -0.1605381
```

### Data Transformations

``` r
scores <- as_mlx(c(0.1, 0.7, 0.4, 0.9))
mlx_sort(scores)
#> mlx array [4]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 0.1 0.4 0.7 0.9
mlx_topk(scores, 2)
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 0.7 0.9
mlx_argmax(scores)
#> mlx array []
#>   dtype: int64
#>   device: gpu
#>   values:
#> [1] 4
```

### Matrix Operations

``` r
A <- as_mlx(matrix(1:4, 2, 2))
B <- as_mlx(matrix(c(1, 0, -1, 0), 2, 2))
A %*% B
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    1   -1
#> [2,]    2   -2
solve(A, B)
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]   -2    2
#> [2,]    1   -1
crossprod(A, B)
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    1   -1
#> [2,]    3   -3
```

### Automatic Differentiation

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
#> [1,] -0.77310860
#> [2,]  0.14119890
#> [3,] -0.12343385
#> [4,]  0.05721617

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
#> [1] 0.06682333
```

## Data Types

Supported data types:

- `float32` for numeric data (default)
- `bool` for logical data
- Integer types `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`,
  `uint32`, `uint64`.
- `complex64`
