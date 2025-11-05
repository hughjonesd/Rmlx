
# Rmlx

<!-- badges: start -->

[![Codecov test
coverage](https://codecov.io/gh/hughjonesd/Rmlx/graph/badge.svg)](https://app.codecov.io/gh/hughjonesd/Rmlx)
<!-- badges: end -->

Rmlx provides an R interface to Apple’s [MLX
framework](https://ml-explore.github.io/mlx/), enabling high-performance
GPU computing on Apple Silicon.

Modern Macs have a GPU, which is great for performing matrix operations.
Statistics uses a lot of matrix operations. But until now, there has
been no way for R on the Mac to use the GPU. Rmlx exists to fill that
gap. It is very early stage and was largely vibe-coded with
Claude/OpenAI Codex. Obviously, use at your own risk! **Contributions
are very welcome.** In particular it would be great to implement
function import/export, more neural network components, etc.

There is a companion library at
[hughjonesd/RmlxStats](https://github.com/hughjonesd/RmlxStats), which
focuses on implementing common statistical methods on the GPU.

Most C++ functions are implemented via R functions with an `mlx_`
prefix. In addition, the package defines mlx-specific methods for many R
matrix operations, including arithmetic, subsetting and matrix algebra.

## Requirements

MacOS on Apple Silicon *or* Linux with CUDA *or* MacOS/Linux for a
CPU-only build.

## Installation

`brew install mlx`

or the Linux equivalent. Then just install the package as normal:

``` r
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
#>     asplit, backsolve, chol2inv, col, colMeans, colSums, diag, drop,
#>     outer, row, rowMeans, rowSums, svd

A <- matrix(rnorm(1e6), 1e3, 1e3)
A_mlx <- as_mlx(A)
system.time(solve(A))
#>    user  system elapsed 
#>   0.378   0.005   0.555
system.time(mlx_eval(solve(A_mlx))) 
#>    user  system elapsed 
#>   0.008   0.006   0.015
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

# Matrix algebra
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
colMeans(a)
#> mlx array [3]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1.5 3.5 5.5

# Cumulative operations flatten column-major
cumsum(a)
#> mlx array [6]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1]  1  3  6 10 15 21

qr_res <- qr(a)
svd_res <- svd(a)
chol_res <- chol(a[, 1:2])
fft_res <- fft(a)
crossprod_res <- crossprod(a, b[1:2, ])
solve_res <- solve(a[, 1:2], b[1:2, ])
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
#>           [,1]
#> [1,] 0.9005783
#> [2,] 0.2985840
#> [3,] 0.6431852
#> [4,] 0.7269748

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
ypred <- mlx_forward(model, x)
mean((ypred - y) * (ypred - y))
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 0.1833142
```

## Learning more

- [Package website](https://hughjonesd.github.io/Rmlx)
- [Apple MLX
  documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [Package
  help](https://hughjonesd.github.io/Rmlx/reference/Rmlx-package.html)
- [Function
  reference](https://hughjonesd.github.io/Rmlx/reference/index.html)
