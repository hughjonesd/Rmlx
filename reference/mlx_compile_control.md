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
#>           [,1]       [,2]       [,3]      [,4]
#> [1,] 2.3701034 -0.3116362  2.0149195 0.7646685
#> [2,] 0.9308830  1.5968647 -0.4051466 1.0937041
#> [3,] 0.9921152  2.0219376  0.2792870 0.5017642
#> [4,] 0.8945001  1.8575875  1.1461381 1.1303174

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]       [,2]       [,3]      [,4]
#> [1,] 2.3701034 -0.3116362  2.0149195 0.7646685
#> [2,] 0.9308830  1.5968647 -0.4051466 1.0937041
#> [3,] 0.9921152  2.0219376  0.2792870 0.5017642
#> [4,] 0.8945001  1.8575875  1.1461381 1.1303174
```
