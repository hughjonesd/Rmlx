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
#>           [,1]       [,2]      [,3]      [,4]
#> [1,] 0.6168116 0.72185469 2.4072938  1.143832
#> [2,] 2.0882635 0.05029684 0.6845186 -0.450691
#> [3,] 2.0121984 1.36850953 2.5208855  2.222666
#> [4,] 1.7007413 1.09995055 2.7126625  1.424645

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>           [,1]       [,2]      [,3]      [,4]
#> [1,] 0.6168116 0.72185469 2.4072938  1.143832
#> [2,] 2.0882635 0.05029684 0.6845186 -0.450691
#> [3,] 2.0121984 1.36850953 2.5208855  2.222666
#> [4,] 1.7007413 1.09995055 2.7126625  1.424645
```
