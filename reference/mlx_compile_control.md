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
#>             [,1]      [,2]       [,3]       [,4]
#> [1,] -0.79283750 0.8670310  1.6697741  0.5936022
#> [2,]  0.04980022 2.2349653 -0.3946849  2.0154319
#> [3,] -0.46392620 0.5415570  1.5175397  3.3943517
#> [4,] -0.26884758 0.4595797  1.3976451 -0.9692717

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]      [,2]       [,3]       [,4]
#> [1,] -0.79283750 0.8670310  1.6697741  0.5936022
#> [2,]  0.04980022 2.2349653 -0.3946849  2.0154319
#> [3,] -0.46392620 0.5415570  1.5175397  3.3943517
#> [4,] -0.26884758 0.4595797  1.3976451 -0.9692717
```
