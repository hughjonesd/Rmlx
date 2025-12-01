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
#> [1,] 1.9532853 -0.4083607 0.3003741  0.3485568
#> [2,] 1.9267871  2.0081241 1.4855127  0.7882814
#> [3,] 0.5005217 -0.5085607 1.6432438 -0.6443727
#> [4,] 3.0798161  0.4967358 1.6016903  0.8984825

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]       [,2]      [,3]       [,4]
#> [1,] 1.9532853 -0.4083607 0.3003741  0.3485568
#> [2,] 1.9267871  2.0081241 1.4855127  0.7882814
#> [3,] 0.5005217 -0.5085607 1.6432438 -0.6443727
#> [4,] 3.0798161  0.4967358 1.6016903  0.8984825
```
