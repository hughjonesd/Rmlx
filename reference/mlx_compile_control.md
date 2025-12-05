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
#>           [,1]       [,2]       [,3]       [,4]
#> [1,] 1.7586933 -0.6602767  3.2575552  1.5236034
#> [2,] 0.1147048  0.2229075  1.0612774  1.0548495
#> [3,] 2.1853287  0.1284322  2.0946631  0.1028204
#> [4,] 0.2823411  0.2958902 -0.7545877 -0.7207159

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]       [,2]       [,3]       [,4]
#> [1,] 1.7586933 -0.6602767  3.2575552  1.5236034
#> [2,] 0.1147048  0.2229075  1.0612774  1.0548495
#> [3,] 2.1853287  0.1284322  2.0946631  0.1028204
#> [4,] 0.2823411  0.2958902 -0.7545877 -0.7207159
```
