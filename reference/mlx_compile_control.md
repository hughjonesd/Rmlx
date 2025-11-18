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
#>            [,1]      [,2]      [,3]      [,4]
#> [1,]  0.5550277 1.0811387 2.5210509  1.233394
#> [2,] -2.5525410 1.7523637 0.2130073  1.066828
#> [3,]  0.3185196 0.6498611 3.4987383 -1.163328
#> [4,]  0.6443735 1.2190855 1.6405356  1.244220

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]      [,3]      [,4]
#> [1,]  0.5550277 1.0811387 2.5210509  1.233394
#> [2,] -2.5525410 1.7523637 0.2130073  1.066828
#> [3,]  0.3185196 0.6498611 3.4987383 -1.163328
#> [4,]  0.6443735 1.2190855 1.6405356  1.244220
```
