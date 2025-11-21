# Getting Started with Rmlx

## Introduction

Apple MLX (Machine Learning eXchange) is Apple’s high-performance array
and machine-learning framework for Apple Silicon, built on top of Metal
for GPU execution and optimized CPU kernels. It offers lazy evaluation,
vectorized math, automatic differentiation, and neural network building
blocks (see the [official MLX
documentation](https://ml-explore.github.io/mlx/) for full details).

`Rmlx` is a thin R layer over MLX that lets you:

- Create MLX tensors from R data
  ([`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md)).
- Run GPU-accelerated math, linear algebra, FFTs, and reductions with
  familiar R syntax.
- Use automatic differentiation
  ([`mlx_grad()`](https://hughjonesd.github.io/Rmlx/reference/mlx_grad.md),
  [`mlx_value_grad()`](https://hughjonesd.github.io/Rmlx/reference/mlx_grad.md))
  for optimization.
- Build simple models with MLX modules and update them using SGD
  helpers.

All heavy computation stays in MLX land; you only copy back to base R
when you call functions like
[`as.matrix()`](https://rdrr.io/r/base/matrix.html).

## System Requirements

Before using Rmlx, ensure MLX is installed:

``` bash
# Using Homebrew (if available)
brew install mlx

# Or build from source
git clone https://github.com/ml-explore/mlx.git
cd mlx && mkdir build && cd build
cmake .. && make && sudo make install
```

## Creating MLX Arrays

Convert R objects to MLX arrays using
[`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md):

``` r
# From a vector
v <- as_mlx(1:10)
print(v)
#> mlx array [10]
#>   dtype: float32
#>   device: gpu
#>   values:
#>  [1]  1  2  3  4  5  6  7  8  9 10

# From a matrix
m <- matrix(1:12, nrow = 3, ncol = 4)
x <- as_mlx(m)
print(x)
#> mlx array [3 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    4    7   10
#> [2,]    2    5    8   11
#> [3,]    3    6    9   12

# Specify the device explicitly (uses GPU if available, CPU otherwise)
x_dev <- as_mlx(m, device = device)
```

> **Precision note:** Numeric inputs are stored in single precision
> (`float32`). Requesting `dtype = "float64"` will downcast the input
> with a warning. Logical inputs are stored as MLX `bool` tensors
> (logical `NA` values are not supported). Complex inputs are stored as
> `complex64` (single-precision real and imaginary parts). Use base R
> arrays if you need double precision arithmetic.

## Lazy Evaluation

MLX arrays use lazy evaluation - operations are recorded but not
computed until needed:

``` r
# These operations are not computed immediately
x <- as_mlx(matrix(1:100, 10, 10))
y <- as_mlx(matrix(101:200, 10, 10))
z <- x + y * 2

# Force evaluation of a specific array
mlx_eval(z)

# Or convert to R (automatically evaluates)
as.matrix(z)
#>       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#>  [1,]  203  233  263  293  323  353  383  413  443   473
#>  [2,]  206  236  266  296  326  356  386  416  446   476
#>  [3,]  209  239  269  299  329  359  389  419  449   479
#>  [4,]  212  242  272  302  332  362  392  422  452   482
#>  [5,]  215  245  275  305  335  365  395  425  455   485
#>  [6,]  218  248  278  308  338  368  398  428  458   488
#>  [7,]  221  251  281  311  341  371  401  431  461   491
#>  [8,]  224  254  284  314  344  374  404  434  464   494
#>  [9,]  227  257  287  317  347  377  407  437  467   497
#> [10,]  230  260  290  320  350  380  410  440  470   500

# Wait for all queued work on the available device
mlx_synchronize(device)
```

## Arithmetic Operations

Rmlx supports standard arithmetic operators:

``` r
x <- as_mlx(matrix(1:12, 3, 4))
y <- as_mlx(matrix(13:24, 3, 4))

# Element-wise operations
sum_xy <- x + y
diff_xy <- x - y
prod_xy <- x * y
quot_xy <- x / y
pow_xy <- x ^ 2

# Convert back to R to see results
as.matrix(sum_xy)
#>      [,1] [,2] [,3] [,4]
#> [1,]   14   20   26   32
#> [2,]   16   22   28   34
#> [3,]   18   24   30   36
```

## Matrix Operations

### Matrix Multiplication

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

### Transpose

``` r
x <- as_mlx(matrix(1:12, 3, 4))
x_t <- t(x)
print(x_t)
#> mlx array [4 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    1    2    3
#> [2,]    4    5    6
#> [3,]    7    8    9
#> [4,]   10   11   12
```

### Cross Products

``` r
x <- as_mlx(matrix(rnorm(20), 5, 4))
true_w <- as_mlx(matrix(c(2, -1, 0.5, 0.25), 4, 1))
y <- x %*% true_w
w <- as_mlx(matrix(0, 4, 1))

# Loss must stay entirely in MLX-land: no conversions back to base R
loss <- function(theta, data_x, data_y) {
  preds <- data_x %*% theta
  resids <- preds - data_y
  sum(resids * resids) / length(data_y)
}

grads <- mlx_grad(loss, w, x, y)

# Wrong: converting to base R breaks the gradient
bad_loss <- function(theta, data_x, data_y) {
  preds <- as.matrix(data_x %*% theta)  # leaves MLX
  resids <- preds - as.matrix(data_y)
  sum(resids * resids) / nrow(resids)
}
try(mlx_grad(bad_loss, w, x, y))
#> Error in eval(expr, envir) : 
#>   MLX autograd failed to differentiate the function: Gradient function must return an `mlx` object. Ensure your closure keeps computations in MLX or wraps the result with as_mlx().
#> Ensure all differentiable computations use MLX operations.

# A small SGD loop using the module/optimizer helpers
model <- mlx_linear(4, 1, bias = FALSE)  # learns a single weight vector
parameters <- mlx_parameters(model)
opt <- mlx_optimizer_sgd(parameters, lr = 0.1)
loss_fn <- function(mod, data_x, data_y) {
  theta <- mlx_param_values(parameters)[[1]]
  loss(theta, data_x, data_y)
}

loss_history <- numeric(50)
for (step in seq_along(loss_history)) {
  step_res <- mlx_train_step(model, loss_fn, opt, x, y)
  loss_history[step] <- as.vector(step_res$loss)
}

# Check final loss and inspect learned parameters
final_loss <- mlx_forward(model, x)
residual_mse <- as.vector(mean((final_loss - y) * (final_loss - y)))
residual_mse
#> [1] 5.658819e-05
loss_history
#>  [1] 8.308889e+00 4.272983e+00 2.288200e+00 1.283615e+00 7.582743e-01
#>  [6] 4.726674e-01 3.102017e-01 2.130744e-01 1.520021e-01 1.117421e-01
#> [11] 8.408680e-02 6.443550e-02 5.009037e-02 3.939489e-02 3.128565e-02
#> [16] 2.505323e-02 2.020860e-02 1.640565e-02 1.339452e-02 1.099170e-02
#> [21] 9.060724e-03 7.498934e-03 6.228307e-03 5.189063e-03 4.335000e-03
#> [26] 3.630075e-03 3.046037e-03 2.560509e-03 2.155664e-03 1.817213e-03
#> [31] 1.533631e-03 1.295554e-03 1.095328e-03 9.267041e-04 7.845016e-04
#> [36] 6.644622e-04 5.630297e-04 4.772589e-04 4.046760e-04 3.432218e-04
#> [41] 2.911653e-04 2.470508e-04 2.096552e-04 1.779418e-04 1.510463e-04
#> [46] 1.282235e-04 1.088608e-04 9.242821e-05 7.848036e-05 6.664058e-05

learned_w <- mlx_param_values(parameters)[[1]]
as.matrix(learned_w)
#>            [,1]
#> [1,]  1.9941052
#> [2,] -1.0081218
#> [3,]  0.4934084
#> [4,]  0.2496413
as.matrix(true_w)
#>       [,1]
#> [1,]  2.00
#> [2,] -1.00
#> [3,]  0.50
#> [4,]  0.25
```

## Reductions

Compute summaries across arrays:

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

# Column and row means
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

# Convert to R to see values
as.matrix(colMeans(x))
#> Warning in as.matrix.mlx(colMeans(x)): Converting array to 1-column matrix
#>       [,1]
#>  [1,]  5.5
#>  [2,] 15.5
#>  [3,] 25.5
#>  [4,] 35.5
#>  [5,] 45.5
#>  [6,] 55.5
#>  [7,] 65.5
#>  [8,] 75.5
#>  [9,] 85.5
#> [10,] 95.5

# Cumulative operations flatten the array in column-major order
as.vector(cumsum(x))
#>   [1]    1    3    6   10   15   21   28   36   45   55   66   78   91  105  120
#>  [16]  136  153  171  190  210  231  253  276  300  325  351  378  406  435  465
#>  [31]  496  528  561  595  630  666  703  741  780  820  861  903  946  990 1035
#>  [46] 1081 1128 1176 1225 1275 1326 1378 1431 1485 1540 1596 1653 1711 1770 1830
#>  [61] 1891 1953 2016 2080 2145 2211 2278 2346 2415 2485 2556 2628 2701 2775 2850
#>  [76] 2926 3003 3081 3160 3240 3321 3403 3486 3570 3655 3741 3828 3916 4005 4095
#>  [91] 4186 4278 4371 4465 4560 4656 4753 4851 4950 5050
```

## Indexing

Subset MLX arrays similar to R:

``` r
x <- as_mlx(matrix(1:100, 10, 10))

# Select rows and columns
x_sub <- x[1:5, 1:5]

# Select specific row
row_1 <- x[1, ]

# Select specific column
col_1 <- x[, 1]
```

## Device Management

Control whether computations run on GPU or CPU:

``` r
# Check default device
mlx_default_device()
#> [1] "gpu"

# Set to CPU for debugging
mlx_default_device("cpu")
#> [1] "cpu"

# Create array on CPU
x_cpu <- as_mlx(matrix(1:12, 3, 4), device = "cpu")

# Set back to best available device
mlx_default_device(device)
#> [1] "gpu"
```

Remember that numeric computations are always performed in `float32`;
the CPU mode is useful when you need to compare against base R or debug
without a GPU.

## Performance Comparison

Here’s a simple timing comparison for large matrix multiplication:

``` r
n <- 1000

# R base
m1 <- matrix(rnorm(n * n), n, n)
m2 <- matrix(rnorm(n * n), n, n)
t1 <- system.time(r_result <- m1 %*% m2)

# MLX
x1 <- as_mlx(m1)
x2 <- as_mlx(m2)
mlx_eval(x1)
mlx_eval(x2)
t2 <- system.time({
  mlx_result <- x1 %*% x2
  mlx_eval(mlx_result)
  final <- as.matrix(mlx_result)
})

cat("Base R:", t1["elapsed"], "seconds\n")
#> Base R: 0.02 seconds
cat("MLX:", t2["elapsed"], "seconds\n")
#> MLX: 0.022 seconds
```

Note: This is an informal comparison, not a rigorous benchmark.
Performance gains depend on array size, operation type, and hardware.

## Best Practices

1.  **Keep data on GPU**: Minimize transfers between R and MLX
2.  **Use lazy evaluation**: Build computation graphs before evaluating
3.  **Batch operations**: Combine operations before forcing evaluation
4.  **Monitor memory**: GPU memory is limited; free unused arrays
5.  **Start with CPU**: Use CPU device for debugging, then switch to GPU

## Limitations

Current limitations in this initial version:

- Apple Silicon only (no Intel Mac or other platforms)
- 2D arrays (matrices) are primary focus
- Limited indexing operations
- No autodiff or gradient computation (planned for future release)
