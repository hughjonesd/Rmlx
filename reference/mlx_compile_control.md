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
#>           [,1]       [,2]      [,3]       [,4]
#> [1,] 0.2687094 -0.2243724 0.1733695  0.6490242
#> [2,] 1.8584695  1.0903594 1.8731097  1.1409801
#> [3,] 0.6793436  3.1429729 1.1897898 -0.6185143
#> [4,] 0.6610410  0.8816355 0.9403894  0.9978358

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]       [,2]      [,3]       [,4]
#> [1,] 0.2687094 -0.2243724 0.1733695  0.6490242
#> [2,] 1.8584695  1.0903594 1.8731097  1.1409801
#> [3,] 0.6793436  3.1429729 1.1897898 -0.6185143
#> [4,] 0.6610410  0.8816355 0.9403894  0.9978358
```
