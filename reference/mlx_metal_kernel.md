# Create a JIT-compiled custom Metal kernel

Wraps MLX's Metal kernel API so R code can define custom GPU kernels
while keeping inputs and outputs as `mlx` arrays.

## Usage

``` r
mlx_metal_kernel(
  name,
  input_names,
  output_names,
  source,
  header = "",
  ensure_row_contiguous = TRUE,
  atomic_outputs = FALSE
)
```

## Arguments

- name:

  Kernel name used in generated Metal code.

- input_names:

  Character vector naming the kernel inputs.

- output_names:

  Character vector naming the kernel outputs.

- source:

  Metal source for the kernel body. MLX generates the function signature
  automatically.

- header:

  Optional Metal source prepended before the generated function.

- ensure_row_contiguous:

  Logical. Should MLX make inputs row-contiguous before launching the
  kernel?

- atomic_outputs:

  Logical. Should output buffers use Metal atomic types?

## Value

A function that executes the compiled kernel and returns one `mlx` array
for a single output or a named list of `mlx` arrays otherwise.

## See also

[`mlx_compile()`](https://hughjonesd.github.io/Rmlx/reference/mlx_compile.md)

[mlx.core.fast.metal_kernel](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html)

[Custom Metal
Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)

## Examples

``` r
if (FALSE) { # \dontrun{
add_one <- mlx_metal_kernel(
  name = "add_one",
  input_names = "inp",
  output_names = "out",
  source = "
    uint elem = thread_position_in_grid.x;
    out[elem] = inp[elem] + (T)1;
  "
)

x <- mlx_cast(as_mlx(1:8), "float32")
y <- add_one(
  inputs = list(x),
  output_shapes = list(c(length(x))),
  output_dtypes = "float32",
  grid = c(length(x), 1L, 1L),
  threadgroup = c(length(x), 1L, 1L),
  template = list(T = "float32")
)
} # }
```
