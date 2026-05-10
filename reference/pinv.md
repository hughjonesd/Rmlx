# Moore-Penrose pseudoinverse for MLX arrays

Moore-Penrose pseudoinverse for MLX arrays

## Usage

``` r
pinv(x, device = NULL)
```

## Arguments

- x:

  An mlx object or coercible matrix.

## Value

An mlx object containing the pseudoinverse.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Create or cast the
operands with `device = "cpu"` explicitly, or pass a `device = "cpu"`
argument. (Passing the argument won't affect the device of any mlx
object returned, just where this particular operation is run.)

## See also

[mlx.linalg.pinv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.pinv)

## Examples

``` r
x <- mlx_matrix(c(1, 2, 3, 4), 2, 2, device = "cpu")
pinv(x)
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: cpu
#>   values:
#>      [,1]       [,2]
#> [1,]   -2  1.5000004
#> [2,]    1 -0.5000001
```
