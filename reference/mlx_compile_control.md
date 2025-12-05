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
#>             [,1]       [,2]        [,3]      [,4]
#> [1,]  1.74629211 -0.1218981 1.428694129 2.2101984
#> [2,]  1.08629560 -0.8082170 1.462495804 0.8686794
#> [3,]  0.32791382  0.5923543 0.845045626 0.5813308
#> [4,] -0.02061546  0.2817030 0.003900051 0.2282491

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]       [,2]        [,3]      [,4]
#> [1,]  1.74629211 -0.1218981 1.428694129 2.2101984
#> [2,]  1.08629560 -0.8082170 1.462495804 0.8686794
#> [3,]  0.32791382  0.5923543 0.845045626 0.5813308
#> [4,] -0.02061546  0.2817030 0.003900051 0.2282491
```
