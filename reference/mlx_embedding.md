# Embedding layer

Maps discrete tokens to continuous vectors.

## Usage

``` r
mlx_embedding(num_embeddings, embedding_dim, device = mlx_default_device())
```

## Arguments

- num_embeddings:

  Size of vocabulary.

- embedding_dim:

  Dimension of embedding vectors.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Default:
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

An `mlx_module` for token embeddings.

## See also

[mlx.nn.Embedding](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Embedding)

## Examples

``` r
set.seed(1)
emb <- mlx_embedding(num_embeddings = 100, embedding_dim = 16)
# Token indices (1-indexed)
tokens <- as_mlx(matrix(c(5, 10, 3, 7), 2, 2))
mlx_forward(emb, tokens)
#> mlx array [2 x 2 x 16]
#>   dtype: float32
#>   device: gpu
#>   (64 elements, not shown)
```
