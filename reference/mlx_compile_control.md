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
#>           [,1]     [,2]      [,3]     [,4]
#> [1,] 1.4761009 1.735369 1.5138775 1.056454
#> [2,] 1.4071999 1.708234 1.3079991 2.758484
#> [3,] 0.9944593 2.116283 1.7708403 1.364751
#> [4,] 0.8309933 2.518596 0.7615991 1.338667

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]     [,2]      [,3]     [,4]
#> [1,] 1.4761009 1.735369 1.5138775 1.056454
#> [2,] 1.4071999 1.708234 1.3079991 2.758484
#> [3,] 0.9944593 2.116283 1.7708403 1.364751
#> [4,] 0.8309933 2.518596 0.7615991 1.338667
```
