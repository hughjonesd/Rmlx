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
#>           [,1]      [,2]      [,3]       [,4]
#> [1,] 0.7243268  1.960966 0.1972181  1.4634970
#> [2,] 1.5994651  1.588675 1.9863768  0.7880453
#> [3,] 0.9711473 -1.034236 0.6674443 -0.3293152
#> [4,] 1.9922726  1.411119 0.6293382  0.6232531

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]      [,2]      [,3]       [,4]
#> [1,] 0.7243268  1.960966 0.1972181  1.4634970
#> [2,] 1.5994651  1.588675 1.9863768  0.7880453
#> [3,] 0.9711473 -1.034236 0.6674443 -0.3293152
#> [4,] 1.9922726  1.411119 0.6293382  0.6232531
```
