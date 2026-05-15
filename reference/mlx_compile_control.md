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
#>            [,1]        [,2]      [,3]      [,4]
#> [1,] 0.03533053  0.57179785 1.5052241 1.4485571
#> [2,] 1.63678062 -0.07840121 1.0150530 0.3449462
#> [3,] 0.53754258 -0.62047780 0.5392287 1.0022386
#> [4,] 1.43951511  1.63209486 0.4429632 2.4120755

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>            [,1]        [,2]      [,3]      [,4]
#> [1,] 0.03533053  0.57179785 1.5052241 1.4485571
#> [2,] 1.63678062 -0.07840121 1.0150530 0.3449462
#> [3,] 0.53754258 -0.62047780 0.5392287 1.0022386
#> [4,] 1.43951511  1.63209486 0.4429632 2.4120755
```
