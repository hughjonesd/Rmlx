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
#>             [,1]      [,2]     [,3]      [,4]
#> [1,] -0.03034937 2.1155972 1.118765 2.1220405
#> [2,]  0.93951094 1.4264135 2.293178 2.3985085
#> [3,]  0.08412987 0.6549987 1.404685 0.3879153
#> [4,] -0.26197994 0.7577977 0.617563 1.7274554

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]      [,2]     [,3]      [,4]
#> [1,] -0.03034937 2.1155972 1.118765 2.1220405
#> [2,]  0.93951094 1.4264135 2.293178 2.3985085
#> [3,]  0.08412987 0.6549987 1.404685 0.3879153
#> [4,] -0.26197994 0.7577977 0.617563 1.7274554
```
