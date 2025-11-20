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
#>            [,1]        [,2]      [,3]       [,4]
#> [1,]  1.1672386  0.78414977 0.3800405 0.01639986
#> [2,] -0.6823403  1.58825970 0.4142097 1.05813551
#> [3,]  2.1644473  2.27320194 0.5383393 0.70012444
#> [4,]  1.5992517 -0.05057645 2.5945415 0.68729031

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]        [,2]      [,3]       [,4]
#> [1,]  1.1672386  0.78414977 0.3800405 0.01639986
#> [2,] -0.6823403  1.58825970 0.4142097 1.05813551
#> [3,]  2.1644473  2.27320194 0.5383393 0.70012444
#> [4,]  1.5992517 -0.05057645 2.5945415 0.68729031
```
