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
#>            [,1]      [,2]      [,3]      [,4]
#> [1,]  0.5355210 0.1491763 0.2878075 1.3358078
#> [2,]  0.6965024 2.0246277 0.1286460 0.4881639
#> [3,]  1.6538981 1.0725193 2.4273553 0.2644017
#> [4,] -0.6322939 3.7483952 0.2845333 2.3817658

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>            [,1]      [,2]      [,3]      [,4]
#> [1,]  0.5355210 0.1491763 0.2878075 1.3358078
#> [2,]  0.6965024 2.0246277 0.1286460 0.4881639
#> [3,]  1.6538981 1.0725193 2.4273553 0.2644017
#> [4,] -0.6322939 3.7483952 0.2845333 2.3817658
```
