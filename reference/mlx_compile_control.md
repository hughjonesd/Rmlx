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
#>            [,1]      [,2]      [,3]        [,4]
#> [1,]  1.4424186 0.7370757 0.1493127 1.797080517
#> [2,]  0.6860240 1.4012371 2.1063156 0.005309284
#> [3,] -0.2607411 0.1138033 1.8978660 2.903797150
#> [4,]  0.4775114 1.3690230 0.7348169 0.814585805

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]      [,3]        [,4]
#> [1,]  1.4424186 0.7370757 0.1493127 1.797080517
#> [2,]  0.6860240 1.4012371 2.1063156 0.005309284
#> [3,] -0.2607411 0.1138033 1.8978660 2.903797150
#> [4,]  0.4775114 1.3690230 0.7348169 0.814585805
```
