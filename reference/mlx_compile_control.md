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
#>             [,1]       [,2]      [,3]       [,4]
#> [1,]  0.04740244  1.2734002 0.5032557 -0.4951659
#> [2,]  0.91003317  2.0071580 0.6674989 -0.3446096
#> [3,]  0.84098691  1.7715443 0.9956514  0.1065971
#> [4,] -0.08755612 -0.9352223 1.7663426  0.2158364

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]       [,2]      [,3]       [,4]
#> [1,]  0.04740244  1.2734002 0.5032557 -0.4951659
#> [2,]  0.91003317  2.0071580 0.6674989 -0.3446096
#> [3,]  0.84098691  1.7715443 0.9956514  0.1065971
#> [4,] -0.08755612 -0.9352223 1.7663426  0.2158364
```
