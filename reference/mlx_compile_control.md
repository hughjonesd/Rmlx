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
#>           [,1]       [,2]       [,3]      [,4]
#> [1,] 1.4374467  2.1681013  0.1093018 0.4560280
#> [2,] 0.6206253  2.4984570  1.4290469 0.2837476
#> [3,] 1.3943496 -0.5103096 -0.6913903 2.6057792
#> [4,] 1.4631748  1.5356199  0.8419050 2.3155804

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]       [,2]       [,3]      [,4]
#> [1,] 1.4374467  2.1681013  0.1093018 0.4560280
#> [2,] 0.6206253  2.4984570  1.4290469 0.2837476
#> [3,] 1.3943496 -0.5103096 -0.6913903 2.6057792
#> [4,] 1.4631748  1.5356199  0.8419050 2.3155804
```
