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
#>          [,1]      [,2]       [,3]       [,4]
#> [1,] 2.447382 0.5265799  1.4982810 -0.2016854
#> [2,] 2.726206 2.5687413 -1.3451512  1.1764386
#> [3,] 1.929950 0.8774352  2.4185457  2.1120853
#> [4,] 1.547736 0.9567233  0.8085563  1.2345401

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>          [,1]      [,2]       [,3]       [,4]
#> [1,] 2.447382 0.5265799  1.4982810 -0.2016854
#> [2,] 2.726206 2.5687413 -1.3451512  1.1764386
#> [3,] 1.929950 0.8774352  2.4185457  2.1120853
#> [4,] 1.547736 0.9567233  0.8085563  1.2345401
```
