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
#>            [,1]        [,2]       [,3]       [,4]
#> [1,] 0.06076652 -0.08024585 -0.7767626  1.1434448
#> [2,] 0.56156796  1.46033931  1.5872290 -0.3569943
#> [3,] 1.32798147  0.55906558  2.4679475  1.2864002
#> [4,] 0.86755902  0.02518928  0.8720412  0.2089872

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>            [,1]        [,2]       [,3]       [,4]
#> [1,] 0.06076652 -0.08024585 -0.7767626  1.1434448
#> [2,] 0.56156796  1.46033931  1.5872290 -0.3569943
#> [3,] 1.32798147  0.55906558  2.4679475  1.2864002
#> [4,] 0.86755902  0.02518928  0.8720412  0.2089872
```
