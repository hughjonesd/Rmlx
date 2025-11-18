# Check if GPU backend is available

Determines whether the GPU backend was compiled and is available.

## Usage

``` r
mlx_has_gpu()
```

## Value

Logical: `TRUE` if GPU is available, `FALSE` if only CPU.

## Examples

``` r
if (mlx_has_gpu()) {
  mlx_synchronize("gpu")
} else {
  mlx_synchronize("cpu")
}
```
