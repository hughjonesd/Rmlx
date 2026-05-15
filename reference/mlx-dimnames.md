# Dimnames and names for MLX arrays

Get or set R-side dimname metadata on `mlx` arrays. Names are stored as
ordinary R metadata on the wrapper and are not written into MLX storage.

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

[`rownames()`](https://rdrr.io/r/base/colnames.html) and
[`colnames()`](https://rdrr.io/r/base/colnames.html) use these
[`dimnames()`](https://rdrr.io/r/base/dimnames.html) methods through
base R's internal generic dispatch.
