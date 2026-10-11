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
#>             [,1]       [,2]       [,3]       [,4]
#> [1,] -0.60684180  1.4386595  0.2634969  0.8858722
#> [2,]  0.04698128 -0.6121119 -1.1390591  0.7468221
#> [3,]  0.71369338  4.7400284  0.5959138 -0.2406631
#> [4,]  3.04904413  1.3411784  0.8583185  1.4622947

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>             [,1]       [,2]       [,3]       [,4]
#> [1,] -0.60684180  1.4386595  0.2634969  0.8858722
#> [2,]  0.04698128 -0.6121119 -1.1390591  0.7468221
#> [3,]  0.71369338  4.7400284  0.5959138 -0.2406631
#> [4,]  3.04904413  1.3411784  0.8583185  1.4622947
```
