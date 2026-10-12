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
#>           [,1]      [,2]      [,3]       [,4]
#> [1,] 0.4420753 0.6896281 1.9039676 -0.3835990
#> [2,] 0.7832894 2.3425944 2.9344468  2.4061062
#> [3,] 1.5618899 1.9292946 1.7493241 -0.1482974
#> [4,] 2.2246056 1.5192277 0.4390357  0.1492159

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>           [,1]      [,2]      [,3]       [,4]
#> [1,] 0.4420753 0.6896281 1.9039676 -0.3835990
#> [2,] 0.7832894 2.3425944 2.9344468  2.4061062
#> [3,] 1.5618899 1.9292946 1.7493241 -0.1482974
#> [4,] 2.2246056 1.5192277 0.4390357  0.1492159
```
