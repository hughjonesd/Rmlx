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
#>             [,1]      [,2]      [,3]      [,4]
#> [1,]  0.01946354 1.1257771 1.1475542 2.3325276
#> [2,] -0.37732053 0.8792441 0.7420309 1.1073631
#> [3,]  1.64967608 1.2864203 1.5351381 1.0732498
#> [4,]  0.17172414 1.4504775 1.5047362 0.4381691

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]      [,2]      [,3]      [,4]
#> [1,]  0.01946354 1.1257771 1.1475542 2.3325276
#> [2,] -0.37732053 0.8792441 0.7420309 1.1073631
#> [3,]  1.64967608 1.2864203 1.5351381 1.0732498
#> [4,]  0.17172414 1.4504775 1.5047362 0.4381691
```
