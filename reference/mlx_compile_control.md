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
#>           [,1]       [,2]       [,3]      [,4]
#> [1,] 1.0091788  1.1650962  1.9556481 2.0205989
#> [2,] 1.0683672  1.4678977  0.9838990 1.9288659
#> [3,] 2.4108946 -0.2612020 -0.1171542 0.5818797
#> [4,] 0.3209581  0.8202846  1.8291271 1.5486931

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>           [,1]       [,2]       [,3]      [,4]
#> [1,] 1.0091788  1.1650962  1.9556481 2.0205989
#> [2,] 1.0683672  1.4678977  0.9838990 1.9288659
#> [3,] 2.4108946 -0.2612020 -0.1171542 0.5818797
#> [4,] 0.3209581  0.8202846  1.8291271 1.5486931
```
