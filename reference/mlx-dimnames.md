# Dimnames and names for MLX arrays

Since version 0.4.0, Rmlx supports character names and dimnames. These
work much like base R: [`names()`](https://rdrr.io/r/base/names.html),
[`rownames()`](https://rdrr.io/r/base/colnames.html),
[`colnames()`](https://rdrr.io/r/base/colnames.html),
[`dimnames()`](https://rdrr.io/r/base/dimnames.html) and associated
setters all work. Dimnames may be `NULL` as a whole, and any individual
dimension's names may be `NULL`.

## Usage

``` r
# S3 method for class 'mlx'
dimnames(x)

# S3 method for class 'mlx'
dimnames(x) <- value

# S3 method for class 'mlx'
names(x)

# S3 method for class 'mlx'
names(x) <- value
```

## Arguments

- x:

  An object.

- value:

  Replacement names or dimnames.

## Value

The requested names, or `x` with updated metadata for replacement forms.

## Details

Dimnames can be convenient, but they also slow down operations by adding
work in base R (mlx can't store character strings). In particular,
slowdowns for subsetting can be considerable, maybe 5x or 6x slower. To
maximize performance, you can avoid names, or remove them with
[`unname()`](https://rdrr.io/r/base/unname.html).
