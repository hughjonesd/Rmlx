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
#>          [,1]       [,2]        [,3]       [,4]
#> [1,] 1.310293 1.82248759  1.06799817  1.1599658
#> [2,] 2.001117 1.47915769 -0.09098673 -0.7450143
#> [3,] 2.407201 0.07233858  3.14665270  0.2071894
#> [4,] 2.266225 1.47345150  2.51769400  0.4172898

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>          [,1]       [,2]        [,3]       [,4]
#> [1,] 1.310293 1.82248759  1.06799817  1.1599658
#> [2,] 2.001117 1.47915769 -0.09098673 -0.7450143
#> [3,] 2.407201 0.07233858  3.14665270  0.2071894
#> [4,] 2.266225 1.47345150  2.51769400  0.4172898
```
