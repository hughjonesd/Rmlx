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
#>            [,1]       [,2]      [,3]      [,4]
#> [1,]  1.2366167 -0.4455465 0.8164317 0.9365245
#> [2,]  0.4025279 -0.4536991 1.0564926 2.1554949
#> [3,] -0.6976330  1.2301693 1.7361779 0.1335049
#> [4,]  0.7017614  1.0699856 1.2686428 1.2183414

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]       [,2]      [,3]      [,4]
#> [1,]  1.2366167 -0.4455465 0.8164317 0.9365245
#> [2,]  0.4025279 -0.4536991 1.0564926 2.1554949
#> [3,] -0.6976330  1.2301693 1.7361779 0.1335049
#> [4,]  0.7017614  1.0699856 1.2686428 1.2183414
```
