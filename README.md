
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

## Requirements

- macOS on Apple Silicon (M1/M2/M3 or later) *or* Linux with CUDA *or*
  macOS/Linux for a CPU-only build
- CMake 3.24 or later
- C++17 compatible compiler

## Installation

Rmlx bundles the MLX library source and builds it automatically during
installation. This typically takes 5-15 minutes on the first install.

### Default Installation

Install from GitHub (builds bundled MLX with optimal backends for your
platform):

``` r
# Using remotes
remotes::install_github("hughjonesd/Rmlx")

# Or using devtools
devtools::install_github("hughjonesd/Rmlx")

# Or from r-universe
install.packages("Rmlx", repos = "https://hughjonesd.r-universe.dev")
```

**macOS (Apple Silicon):** Builds Metal + CPU backends **Linux with
CUDA:** Builds CUDA + CPU backends (requires CUDA toolkit) **Linux
without CUDA:** Builds CPU-only backend

### Custom Backend Configuration

Control which backends to build using `configure.args`:

``` r
# CPU-only build (no GPU acceleration)
remotes::install_github("hughjonesd/Rmlx",
                        configure.args = "--cpu-only")

# Force CUDA on Linux
remotes::install_github("hughjonesd/Rmlx",
                        configure.args = "--with-cuda")

# Disable Metal on macOS
remotes::install_github("hughjonesd/Rmlx",
                        configure.args = "--without-metal")
```

Or use environment variables:

``` r
Sys.setenv(MLX_BUILD_CPU = "ON", MLX_BUILD_CUDA = "OFF")
remotes::install_github("hughjonesd/Rmlx")
```

### Using System-Installed MLX

If you have MLX installed separately (e.g., via Homebrew), you can skip
the bundled build:

``` r
# Auto-detect system MLX
Sys.setenv(MLX_USE_SYSTEM = "1")
remotes::install_github("hughjonesd/Rmlx")

# Or specify paths explicitly
Sys.setenv(MLX_INCLUDE = "/opt/homebrew/include")
Sys.setenv(MLX_LIB_DIR = "/opt/homebrew/lib")
remotes::install_github("hughjonesd/Rmlx")
```

To install system MLX on macOS:

``` bash
brew install mlx
```

See the [INSTALL](INSTALL) file for detailed platform-specific
instructions and troubleshooting.

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
#>   0.387   0.003   0.392
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
x <- as_mlx(matrix(1:100, 10, 10))
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

# drop = FALSE by default
x[1, ]
#> mlx array [1 x 10]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#> [1,]    1   11   21   31   41   51   61   71   81    91

logical_mask <- rep(c(TRUE, FALSE), 5)
x[logical_mask, ]
#> mlx array [5 x 10]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#> [1,]    1   11   21   31   41   51   61   71   81    91
#> [2,]    3   13   23   33   43   53   63   73   83    93
#> [3,]    5   15   25   35   45   55   65   75   85    95
#> [4,]    7   17   27   37   47   57   67   77   87    97
#> [5,]    9   19   29   39   49   59   69   79   89    99
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
c <- a %*% b
as.matrix(c)
#>      [,1] [,2]
#> [1,]   22   49
#> [2,]   28   64

# Reductions
x <- as_mlx(matrix(1:25, 5, 5))

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
mlx_rand_uniform(c(512, 512), min = -1, max = 1)
#> mlx array [512 x 512]
#>   dtype: float32
#>   device: gpu
#>   (262144 elements, not shown)
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
#>   dtype: uint32
#>   device: gpu
#>   values:
#> [1] 3
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
#>            [,1]
#> [1,]  0.4909797
#> [2,]  0.2869464
#> [3,]  1.9843374
#> [4,] -0.6045319

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
#> [1] 0.5065953
```

## Data Types

Supported data types:

- `float32` for numeric data (default)
- `bool` for logical data
- Integer types `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`,
  `uint32`, `uint64`.
- `complex64`

``` r

x_f32 <- as_mlx(matrix(1:12, 3, 4), dtype = "float32")
logical_mat <- as_mlx(matrix(c(TRUE, FALSE, TRUE, TRUE), 2, 2))

# Integer matrix must be requested explicitly:
typeof(1:10)
#> [1] "integer"
x_float <- as_mlx(1:10)
x_int <- as_mlx(1:10, dtype = "int32")

# The Apple GPU uses float32 internally. Requests for `dtype = "float64"` 
# are downcast with a warning.
as_mlx(matrix(1:12, 3, 4), dtype = "float64")
#> Warning: MLX arrays are stored in float32; downcasting input.
#> mlx array [3 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    4    7   10
#> [2,]    2    5    8   11
#> [3,]    3    6    9   12
```
