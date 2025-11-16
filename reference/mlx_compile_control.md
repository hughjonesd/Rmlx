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
#>           [,1]       [,2]      [,3]      [,4]
#> [1,] 0.2431504 0.91247684 0.8288532 1.4930767
#> [2,] 1.3523350 1.43996799 0.7530431 1.1429410
#> [3,] 2.5443821 0.07821369 2.0171211 0.6599144
#> [4,] 0.9184956 2.60200167 1.9060806 0.5565097

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]       [,2]      [,3]      [,4]
#> [1,] 0.2431504 0.91247684 0.8288532 1.4930767
#> [2,] 1.3523350 1.43996799 0.7530431 1.1429410
#> [3,] 2.5443821 0.07821369 2.0171211 0.6599144
#> [4,] 0.9184956 2.60200167 1.9060806 0.5565097
```
