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
#>            [,1]      [,2]      [,3]      [,4]
#> [1,]  1.0885874 0.7768625 0.9621620 2.3668942
#> [2,]  1.2404819 0.5184271 2.4920807 0.8550780
#> [3,]  0.2689133 0.3263482 1.4498489 0.1389729
#> [4,] -0.4080561 1.2234949 0.8776785 2.3516393

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>            [,1]      [,2]      [,3]      [,4]
#> [1,]  1.0885874 0.7768625 0.9621620 2.3668942
#> [2,]  1.2404819 0.5184271 2.4920807 0.8550780
#> [3,]  0.2689133 0.3263482 1.4498489 0.1389729
#> [4,] -0.4080561 1.2234949 0.8776785 2.3516393
```
