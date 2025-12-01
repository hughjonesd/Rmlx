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
#>            [,1]      [,2]      [,3]        [,4]
#> [1,]  0.5412496 1.3121715 1.8609107 -0.73426545
#> [2,]  0.2240267 2.1389894 1.9429014  0.66163743
#> [3,] -0.3465536 1.5817591 0.7233620 -1.28213096
#> [4,]  0.8842950 0.6434481 0.2729192 -0.07692087

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]      [,3]        [,4]
#> [1,]  0.5412496 1.3121715 1.8609107 -0.73426545
#> [2,]  0.2240267 2.1389894 1.9429014  0.66163743
#> [3,] -0.3465536 1.5817591 0.7233620 -1.28213096
#> [4,]  0.8842950 0.6434481 0.2729192 -0.07692087
```
