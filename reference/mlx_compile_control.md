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
#> [1,] 3.1042495 -1.5137591  0.5458150 0.6002162
#> [2,] 2.2795479  0.9259114  2.1075258 1.0985661
#> [3,] 0.4815078  2.2215121  0.4904882 1.6691973
#> [4,] 0.7191830 -0.1229018 -0.2756373 0.8237527

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>           [,1]       [,2]       [,3]      [,4]
#> [1,] 3.1042495 -1.5137591  0.5458150 0.6002162
#> [2,] 2.2795479  0.9259114  2.1075258 1.0985661
#> [3,] 0.4815078  2.2215121  0.4904882 1.6691973
#> [4,] 0.7191830 -0.1229018 -0.2756373 0.8237527
```
