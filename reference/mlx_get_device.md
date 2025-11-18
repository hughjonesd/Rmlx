# Get best available device

Returns `"gpu"` if available, otherwise `"cpu"`.

## Usage

``` r
mlx_get_device()
```

## Value

Character: `"gpu"` or `"cpu"`.

## Examples

``` r
device <- mlx_get_device()
x <- as_mlx(1:10, device = device)
```
