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
#>          [,1]       [,2]        [,3]      [,4]
#> [1,] 2.130244  1.7928095  2.02967691 0.4180322
#> [2,] 1.447663 -1.1681716 -0.03973413 0.1083277
#> [3,] 1.300003  1.7223109  0.77630508 0.1059468
#> [4,] 1.077636 -0.3420846  1.84491456 1.2992934

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>          [,1]       [,2]        [,3]      [,4]
#> [1,] 2.130244  1.7928095  2.02967691 0.4180322
#> [2,] 1.447663 -1.1681716 -0.03973413 0.1083277
#> [3,] 1.300003  1.7223109  0.77630508 0.1059468
#> [4,] 1.077636 -0.3420846  1.84491456 1.2992934
```
