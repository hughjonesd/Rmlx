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
#>            [,1]       [,2]        [,3]      [,4]
#> [1,]  1.4333031 -1.8294582  0.73156393 1.2406770
#> [2,]  0.4542711  0.3660778 -0.03956842 1.7268374
#> [3,]  1.9159989  1.1630547  0.27273345 1.5935296
#> [4,] -1.6001940  0.4171806  0.57940876 0.8591227

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]       [,2]        [,3]      [,4]
#> [1,]  1.4333031 -1.8294582  0.73156393 1.2406770
#> [2,]  0.4542711  0.3660778 -0.03956842 1.7268374
#> [3,]  1.9159989  1.1630547  0.27273345 1.5935296
#> [4,] -1.6001940  0.4171806  0.57940876 0.8591227
```
