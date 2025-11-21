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
#> [1,]  2.5223603 3.0528362 2.0727289 1.7479630
#> [2,]  0.7436332 0.6326073 0.9114606 0.4038348
#> [3,]  1.0483581 2.3535733 0.9180566 1.8157644
#> [4,] -0.7133169 2.2323365 1.8833240 2.1054773

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]      [,3]      [,4]
#> [1,]  2.5223603 3.0528362 2.0727289 1.7479630
#> [2,]  0.7436332 0.6326073 0.9114606 0.4038348
#> [3,]  1.0483581 2.3535733 0.9180566 1.8157644
#> [4,] -0.7133169 2.2323365 1.8833240 2.1054773
```
