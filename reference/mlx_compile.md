# Compile an MLX Function for Optimized Execution

Returns a compiled version of a function that traces and optimizes the
computation graph on first call, then reuses the compiled graph for
subsequent calls with matching input shapes and types.

## Usage

``` r
mlx_compile(f, shapeless = FALSE)
```

## Arguments

- f:

  An R function that takes MLX arrays as arguments and returns MLX
  array(s). The function must be pure (no side effects) and use only MLX
  operations.

- shapeless:

  Logical. If `TRUE`, the compiled function won't recompile when input
  shapes change. However, changing input dtypes or number of dimensions
  still triggers recompilation. Default: `FALSE`

## Value

A compiled function with the same signature as `f`. The first call will
be slow (tracing and compilation), but subsequent calls will be much
faster.

## Details

### How Compilation Works

When you call `mlx_compile(f)`, it returns a new function immediately
without any tracing. The actual compilation happens on the **first
call** to the compiled function:

1.  **First call**: MLX traces the function with placeholder inputs,
    builds the computation graph, optimizes it (fusing operations,
    eliminating redundancy), and caches the result. This is slow.

2.  **Subsequent calls**: If inputs have the same shapes and dtypes, MLX
    reuses the cached compiled graph. This is fast.

3.  **Recompilation**: Occurs when input shapes change (unless
    `shapeless = TRUE`), input dtypes change, or the number of arguments
    changes.

### Requirements for Compiled Functions

Your function must:

- Accept only MLX arrays as arguments

- Return MLX array(s) - either a single mlx object or a list of mlx
  objects

- Use only MLX operations (no conversion to R)

- Be pure (no side effects, no external state modification)

Your function **cannot**:

- Print or evaluate arrays during execution
  ([`print()`](https://rdrr.io/r/base/print.html),
  [`as.matrix()`](https://rdrr.io/r/base/matrix.html),
  [`as.numeric()`](https://rdrr.io/r/base/numeric.html), `[[`
  extraction, etc.)

- Use control flow based on array values (`if (x > 0)` where `x` is an
  array)

- Modify external variables or have other side effects

- Return non-MLX values

### Performance Benefits

- **Operation fusion**: Combines multiple operations into optimized
  kernels

- **Memory reduction**: Eliminates intermediate allocations

- **Overhead reduction**: Bypasses R/C++ call overhead for fused
  operations

Typical speedups range from 2-10x for operation-heavy functions.

### Shapeless Compilation

Setting `shapeless = TRUE` allows the compiled function to handle
varying input shapes without recompilation:

    # Regular compilation - recompiles for each new shape
    fast_fn <- mlx_compile(matmul_fn)
    fast_fn(mlx_zeros(c(10, 64)), weights)  # Compiles for shape (10, 64)
    fast_fn(mlx_zeros(c(20, 64)), weights)  # Recompiles for shape (20, 64)

    # Shapeless compilation - compiles once
    fast_fn <- mlx_compile(matmul_fn, shapeless = TRUE)
    fast_fn(mlx_zeros(c(10, 64)), weights)  # Compiles once
    fast_fn(mlx_zeros(c(20, 64)), weights)  # No recompilation!

Shapeless mode sacrifices some optimization opportunities but avoids
recompilation costs. Use it when processing variable-sized batches.

## See also

[`mlx_disable_compile()`](https://hughjonesd.github.io/Rmlx/reference/mlx_compile_control.md),
[`mlx_enable_compile()`](https://hughjonesd.github.io/Rmlx/reference/mlx_compile_control.md)

[mlx.core.compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html)

## Examples

``` r
# Simple example
matmul_add <- function(x, w, b) {
  (x %*% w) + b
}

# Compile it (returns immediately, no tracing yet)
fast_fn <- mlx_compile(matmul_add)

# First call: slow (traces and compiles)
x <- mlx_rand_normal(c(32, 128))
w <- mlx_rand_normal(c(128, 256))
b <- mlx_rand_normal(c(256))
result <- fast_fn(x, w, b)  # Compiles during this call

# Subsequent calls: fast (uses cached graph)
for (i in 1:1000) {
  result <- fast_fn(batch_data[[i]], w, b)  # Very fast!
}
#> Error: object 'batch_data' not found

# Multiple returns
forward_and_norm <- function(x, w) {
  y <- x %*% w
  norm <- sqrt(sum(y * y))
  list(y, norm)  # Return list of mlx objects
}

compiled_fn <- mlx_compile(forward_and_norm)
results <- compiled_fn(x, w)  # Returns list(y, norm)
```
