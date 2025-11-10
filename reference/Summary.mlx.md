# Summary operations for MLX arrays

S3 group generic for summary functions including
[`sum()`](https://rdrr.io/r/base/sum.html),
[`prod()`](https://rdrr.io/r/base/prod.html),
[`min()`](https://rdrr.io/r/base/Extremes.html),
[`max()`](https://rdrr.io/r/base/Extremes.html),
[`all()`](https://rdrr.io/r/base/all.html), and
[`any()`](https://rdrr.io/r/base/any.html).

## Usage

``` r
# S3 method for class 'mlx'
Summary(x, ..., na.rm = FALSE)
```

## Arguments

- x:

  mlx array or object coercible to mlx

- ...:

  Additional mlx arrays (for reducing multiple arrays), or named
  arguments `axis` and `drop`

- na.rm:

  Logical; currently ignored for mlx arrays (generates warning if TRUE)

## Value

An mlx array with the summary result

## See also

[mlx.core.array](https://ml-explore.github.io/mlx/build/html/python/array.html)

## Examples

``` r
x <- as_mlx(matrix(1:6, 2, 3))
sum(x)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 21
any(x > 3)
#> [1] TRUE
all(x > 0)
#> [1] TRUE
```
