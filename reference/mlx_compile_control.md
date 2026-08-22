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
#>            [,1]     [,2]      [,3]       [,4]
#> [1,] -0.9416795 2.153163 0.2068369  1.1936110
#> [2,]  0.2642437 1.337970 0.5944995  1.4763327
#> [3,]  0.8852443 1.657790 2.0199463  0.9198688
#> [4,]  1.7283669 1.012823 1.9935549 -0.2929045

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>            [,1]     [,2]      [,3]       [,4]
#> [1,] -0.9416795 2.153163 0.2068369  1.1936110
#> [2,]  0.2642437 1.337970 0.5944995  1.4763327
#> [3,]  0.8852443 1.657790 2.0199463  0.9198688
#> [4,]  1.7283669 1.012823 1.9935549 -0.2929045
```
