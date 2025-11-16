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
#>            [,1]       [,2]      [,3]     [,4]
#> [1,]  0.4238577  0.8317425 1.9296093 1.935566
#> [2,]  1.5362611  2.0769181 0.5099679 1.423758
#> [3,]  1.2464325  1.9753976 1.4489200 0.988144
#> [4,] -0.4151329 -1.0589223 1.2554133 1.215293

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]       [,2]      [,3]     [,4]
#> [1,]  0.4238577  0.8317425 1.9296093 1.935566
#> [2,]  1.5362611  2.0769181 0.5099679 1.423758
#> [3,]  1.2464325  1.9753976 1.4489200 0.988144
#> [4,] -0.4151329 -1.0589223 1.2554133 1.215293
```
