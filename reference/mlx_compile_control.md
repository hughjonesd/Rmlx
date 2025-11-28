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
#>            [,1]      [,2]      [,3]       [,4]
#> [1,]  0.9489936 0.0121485 1.8436117  1.3729246
#> [2,]  0.8905634 1.3919261 0.2309837  0.0420298
#> [3,] -1.9327507 1.1992047 1.2587173 -0.1531683
#> [4,]  2.6509333 1.0102551 1.2031807  1.0973016

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]      [,3]       [,4]
#> [1,]  0.9489936 0.0121485 1.8436117  1.3729246
#> [2,]  0.8905634 1.3919261 0.2309837  0.0420298
#> [3,] -1.9327507 1.1992047 1.2587173 -0.1531683
#> [4,]  2.6509333 1.0102551 1.2031807  1.0973016
```
