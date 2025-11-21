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
#>           [,1]      [,2]       [,3]      [,4]
#> [1,] 1.1295974 1.0916111  1.6771584 1.6331582
#> [2,] 2.4097581 1.5185785  0.6878117 1.2480717
#> [3,] 0.7982219 2.7596941  0.4708849 2.1246407
#> [4,] 0.6545485 0.2188222 -0.5512208 0.9171428

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]      [,2]       [,3]      [,4]
#> [1,] 1.1295974 1.0916111  1.6771584 1.6331582
#> [2,] 2.4097581 1.5185785  0.6878117 1.2480717
#> [3,] 0.7982219 2.7596941  0.4708849 2.1246407
#> [4,] 0.6545485 0.2188222 -0.5512208 0.9171428
```
