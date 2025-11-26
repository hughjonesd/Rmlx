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
#>            [,1]      [,2]       [,3]
#> [1,]  3.4694786 -1.947971  0.7083434
#> [2,] -0.1372439  2.180840 -0.1373530
#> [3,] -0.8607467  1.066743  1.3697486
```
