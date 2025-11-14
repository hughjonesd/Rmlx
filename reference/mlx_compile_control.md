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
#>          [,1]     [,2]     [,3]       [,4]
#> [1,] 1.715520 1.242361 1.996333 -0.4238644
#> [2,] 3.655209 4.293869 1.518245  0.7377305
#> [3,] 1.164643 1.009010 1.542184 -0.1734242
#> [4,] 2.187485 1.621400 1.893821 -0.5893197

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>          [,1]     [,2]     [,3]       [,4]
#> [1,] 1.715520 1.242361 1.996333 -0.4238644
#> [2,] 3.655209 4.293869 1.518245  0.7377305
#> [3,] 1.164643 1.009010 1.542184 -0.1734242
#> [4,] 2.187485 1.621400 1.893821 -0.5893197
```
