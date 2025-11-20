# Fused matrix multiply and add for MLX arrays

Computes `beta * input + alpha * (mat1 %*% mat2)` in a single MLX
kernel. All operands are promoted to a common dtype/device prior to
evaluation.

## Usage

``` r
mlx_addmm(input, mat1, mat2, alpha = 1, beta = 1)
```

## Arguments

- input:

  Matrix-like object providing the additive term.

- mat1:

  Left matrix operand.

- mat2:

  Right matrix operand.

- alpha, beta:

  Numeric scalars controlling the fused linear combination.

## Value

An `mlx` matrix with the same shape as `input`.

## See also

[mlx.core.addmm](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.addmm)

## Examples

``` r
input <- as_mlx(diag(3))
mat1 <- as_mlx(matrix(rnorm(9), 3, 3))
mat2 <- as_mlx(matrix(rnorm(9), 3, 3))
mlx_addmm(input, mat1, mat2, alpha = 0.5, beta = 2)
#> mlx array [3 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]      [,3]
#> [1,]  2.3344519 0.4014955 -0.607805
#> [2,] -0.7194482 0.9909397  1.389078
#> [3,]  0.0781863 0.6478112  1.663081
```
