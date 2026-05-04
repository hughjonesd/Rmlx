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
#>            [,1]        [,2]        [,3]      [,4]
#> [1,]  1.3719997  0.89314246  0.23911577  2.425891
#> [2,] -0.8618355  1.72621572 -0.94034195 -0.122930
#> [3,]  0.6977049  1.67414248  0.12098229  1.373723
#> [4,]  2.3304434 -0.01530302  0.04762107  1.688744

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]        [,2]        [,3]      [,4]
#> [1,]  1.3719997  0.89314246  0.23911577  2.425891
#> [2,] -0.8618355  1.72621572 -0.94034195 -0.122930
#> [3,]  0.6977049  1.67414248  0.12098229  1.373723
#> [4,]  2.3304434 -0.01530302  0.04762107  1.688744
```
