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
#>            [,1]        [,2]       [,3]       [,4]
#> [1,]  0.9041877 -0.72551131  0.5847347  1.1341189
#> [2,]  2.3796825 -0.65653324  0.9357227 -0.9038148
#> [3,] -0.4619117  1.87072158  2.0903161  0.6840420
#> [4,]  1.1402029  0.07379115 -0.1607413  1.6223562

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]        [,2]       [,3]       [,4]
#> [1,]  0.9041877 -0.72551131  0.5847347  1.1341189
#> [2,]  2.3796825 -0.65653324  0.9357227 -0.9038148
#> [3,] -0.4619117  1.87072158  2.0903161  0.6840420
#> [4,]  1.1402029  0.07379115 -0.1607413  1.6223562
```
