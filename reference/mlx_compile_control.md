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
#>             [,1]      [,2]        [,3]       [,4]
#> [1,]  1.80421495 1.8997455 -0.62243533 -0.1510921
#> [2,]  0.03577799 2.2571011  0.63187629  0.7051331
#> [3,]  0.50611222 3.5867679  0.08777118 -0.7134863
#> [4,] -1.66743684 0.6260061  2.23889637  0.1084813

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]      [,2]        [,3]       [,4]
#> [1,]  1.80421495 1.8997455 -0.62243533 -0.1510921
#> [2,]  0.03577799 2.2571011  0.63187629  0.7051331
#> [3,]  0.50611222 3.5867679  0.08777118 -0.7134863
#> [4,] -1.66743684 0.6260061  2.23889637  0.1084813
```
