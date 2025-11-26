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
#>            [,1]      [,2]       [,3]       [,4]
#> [1,]  3.7055242 1.7974685 1.67879891  1.5675011
#> [2,]  1.7036710 0.3785902 1.33923054 -0.7131801
#> [3,] -0.1376321 1.0577189 0.77395290  1.5488822
#> [4,]  0.9153895 1.1739984 0.01591831  0.6398693

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]       [,3]       [,4]
#> [1,]  3.7055242 1.7974685 1.67879891  1.5675011
#> [2,]  1.7036710 0.3785902 1.33923054 -0.7131801
#> [3,] -0.1376321 1.0577189 0.77395290  1.5488822
#> [4,]  0.9153895 1.1739984 0.01591831  0.6398693
```
