# Import an exported MLX function

Loads a function previously exported with the MLX Python utilities and
returns an R callable.

## Usage

``` r
mlx_import_function(path, device = mlx_default_device())
```

## Arguments

- path:

  Path to a `.mlxfn` file created via MLX export utilities.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

An R function. Calling it returns an `mlx` array if the imported
function has a single output, or a list of `mlx` arrays otherwise.

## Details

Imported functions behave like regular R closures:

- Positional arguments are passed first and become the positional inputs
  the original MLX function expects.

- Named arguments (e.g. `bias = ...`) become MLX keyword arguments and
  must match the names that were used when exporting.

- Each argument is coerced to `mlx` via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md)
  and automatically moved to the requested device/stream before
  execution.

- If the MLX function yields a single array the result is returned as an
  `mlx` object; multiple outputs are returned as a list in the order MLX
  produced them.

Because `.mlxfn` files can bundle multiple traces (different shapes or
keyword combinations), the imported callable keeps a varargs (`...`)
signature. MLX selects the appropriate trace at runtime based on the
shapes and keyword names you provide.

## Examples

``` r
add_fn <- mlx_import_function(
  system.file("extdata/add_matrix.mlxfn", package = "Rmlx"),
  device = "cpu"
)
x <- mlx_matrix(1:4, 2, 2)
y <- mlx_matrix(5:8, 2, 2)
add_fn(x, bias = y)  # positional + keyword argument
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: cpu
#>   values:
#>      [,1] [,2]
#> [1,]    6   10
#> [2,]    8   12
```
