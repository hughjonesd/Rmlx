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
#>           [,1]      [,2]       [,3]        [,4]
#> [1,] 1.3399186 1.4473509  1.6121761  0.79583836
#> [2,] 1.7738423 1.9639907  1.3133721 -0.01952827
#> [3,] 1.0436198 1.4114574 -1.0537603  1.05533087
#> [4,] 0.3337536 0.7091076  0.4560356  0.50952703

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]      [,2]       [,3]        [,4]
#> [1,] 1.3399186 1.4473509  1.6121761  0.79583836
#> [2,] 1.7738423 1.9639907  1.3133721 -0.01952827
#> [3,] 1.0436198 1.4114574 -1.0537603  1.05533087
#> [4,] 0.3337536 0.7091076  0.4560356  0.50952703
```
