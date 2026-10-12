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
#>            [,1]      [,2]      [,3]        [,4]
#> [1,] 0.09309149 0.5268587 0.5854309  0.03583413
#> [2,] 0.77853334 0.5091628 1.3353561 -0.24063754
#> [3,] 0.53303605 0.3729988 2.2416072  2.40417576
#> [4,] 1.90621090 0.1247161 2.5796301  1.79372370

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>            [,1]      [,2]      [,3]        [,4]
#> [1,] 0.09309149 0.5268587 0.5854309  0.03583413
#> [2,] 0.77853334 0.5091628 1.3353561 -0.24063754
#> [3,] 0.53303605 0.3729988 2.2416072  2.40417576
#> [4,] 1.90621090 0.1247161 2.5796301  1.79372370
```
