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
#>            [,1]     [,2]       [,3]      [,4]
#> [1,] -0.4041119 1.395267  1.6635094 0.5713090
#> [2,]  1.2993321 1.389233  1.4650195 0.9815674
#> [3,]  0.7207034 3.407233 -0.2965192 1.4566098
#> [4,]  0.6132213 1.679594  0.2562531 0.7234428

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>            [,1]     [,2]       [,3]      [,4]
#> [1,] -0.4041119 1.395267  1.6635094 0.5713090
#> [2,]  1.2993321 1.389233  1.4650195 0.9815674
#> [3,]  0.7207034 3.407233 -0.2965192 1.4566098
#> [4,]  0.6132213 1.679594  0.2562531 0.7234428
```
