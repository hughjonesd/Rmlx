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
#>           [,1]     [,2]       [,3]     [,4]
#> [1,] 0.9709225 1.372318 -0.9068913 1.523601
#> [2,] 2.8508837 1.402091  1.6799355 1.442919
#> [3,] 0.6998831 1.255361  1.4760609 1.534864
#> [4,] 0.3551850 2.831096  0.8475254 1.954701

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]     [,2]       [,3]     [,4]
#> [1,] 0.9709225 1.372318 -0.9068913 1.523601
#> [2,] 2.8508837 1.402091  1.6799355 1.442919
#> [3,] 0.6998831 1.255361  1.4760609 1.534864
#> [4,] 0.3551850 2.831096  0.8475254 1.954701
```
