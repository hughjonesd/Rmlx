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
#>            [,1]      [,2]       [,3]        [,4]
#> [1,] -0.1748414 1.8196080  0.4212584  1.09888673
#> [2,]  0.4202884 1.8947711 -0.4525054  0.52865392
#> [3,]  0.9654825 0.9852257  0.6093072  2.30658984
#> [4,]  0.2123702 1.2149173  1.0961165 -0.07484901

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]       [,3]        [,4]
#> [1,] -0.1748414 1.8196080  0.4212584  1.09888673
#> [2,]  0.4202884 1.8947711 -0.4525054  0.52865392
#> [3,]  0.9654825 0.9852257  0.6093072  2.30658984
#> [4,]  0.2123702 1.2149173  1.0961165 -0.07484901
```
