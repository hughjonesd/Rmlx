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
#>            [,1]        [,2]      [,3]     [,4]
#> [1,]  3.3612816 -0.22731709 0.7036227 1.766155
#> [2,] -0.5826616  1.20561743 1.3454726 1.878584
#> [3,]  1.2032332  0.61335886 0.9328077 1.329408
#> [4,]  0.9853749 -0.03180671 1.1694496 1.801041

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>            [,1]        [,2]      [,3]     [,4]
#> [1,]  3.3612816 -0.22731709 0.7036227 1.766155
#> [2,] -0.5826616  1.20561743 1.3454726 1.878584
#> [3,]  1.2032332  0.61335886 0.9328077 1.329408
#> [4,]  0.9853749 -0.03180671 1.1694496 1.801041
```
