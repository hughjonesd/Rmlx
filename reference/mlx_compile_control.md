# Control Global Compilation Behavior

- `mlx_disable_compile()` prevents all compilation globally. Compiled
  functions will execute without optimization.

- `mlx_enable_compile()` enables compilation (overrides the
  `MLX_DISABLE_COMPILE` environment variable).

## Usage

``` r
mlx_disable_compile()

mlx_enable_compile()
```

## Value

Invisibly returns `NULL`.

## Details

These functions control whether MLX compilation is enabled globally.

These are useful for debugging (to check if compilation is causing
issues) or benchmarking (to measure compilation overhead vs speedup).

You can also disable compilation by setting the `MLX_DISABLE_COMPILE`
environment variable before loading the package.

## Examples

``` r
demo_fn <- mlx_compile(function(x) x + 1)
x <- mlx_rand_normal(c(4, 4))

# Disable compilation for debugging
mlx_disable_compile()
demo_fn(x)  # Runs without optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]       [,3]       [,4]
#> [1,] -0.3430775 1.0031196  2.2007668  0.1679093
#> [2,]  1.2927899 0.9077863 -1.3516312  1.8386774
#> [3,]  1.8079131 2.2274361 -0.7559493 -1.6256909
#> [4,]  1.1766114 1.2348357  0.7008754  0.6441954

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]       [,3]       [,4]
#> [1,] -0.3430775 1.0031196  2.2007668  0.1679093
#> [2,]  1.2927899 0.9077863 -1.3516312  1.8386774
#> [3,]  1.8079131 2.2274361 -0.7559493 -1.6256909
#> [4,]  1.1766114 1.2348357  0.7008754  0.6441954
```
