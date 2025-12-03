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
#>           [,1]       [,2]     [,3]      [,4]
#> [1,] 1.1597893  1.8387542 1.569091 1.4420836
#> [2,] 0.5812312 -0.3150022 1.660200 0.4975023
#> [3,] 1.7424247  1.1105443 1.102660 1.7297049
#> [4,] 0.5217475  2.6589704 1.075843 0.2768393

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]       [,2]     [,3]      [,4]
#> [1,] 1.1597893  1.8387542 1.569091 1.4420836
#> [2,] 0.5812312 -0.3150022 1.660200 0.4975023
#> [3,] 1.7424247  1.1105443 1.102660 1.7297049
#> [4,] 0.5217475  2.6589704 1.075843 0.2768393
```
