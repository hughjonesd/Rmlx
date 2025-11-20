# Get best available device

Returns `"gpu"` if available, otherwise `"cpu"`.

## Usage

``` r
mlx_best_device()
```

## Value

Character: `"gpu"` or `"cpu"`.

## Examples

``` r
device <- mlx_best_device()
x <- as_mlx(1:10, device = device)
```
