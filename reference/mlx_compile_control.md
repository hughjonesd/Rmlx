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
#>   values:
#>            [,1]      [,2]      [,3]       [,4]
#> [1,]  2.0916083 2.7095990 1.4730461  1.6447330
#> [2,] -0.0472275 0.3821474 0.5999246 -0.3986514
#> [3,]  1.0424341 1.7093747 0.6592008  1.2551405
#> [4,]  1.5504189 0.5042913 1.5089076 -0.1283283

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>            [,1]      [,2]      [,3]       [,4]
#> [1,]  2.0916083 2.7095990 1.4730461  1.6447330
#> [2,] -0.0472275 0.3821474 0.5999246 -0.3986514
#> [3,]  1.0424341 1.7093747 0.6592008  1.2551405
#> [4,]  1.5504189 0.5042913 1.5089076 -0.1283283
```
