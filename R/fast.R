#' Create a JIT-compiled custom Metal kernel
#'
#' Wraps MLX's Metal kernel API so R code can define custom GPU kernels while
#' keeping inputs and outputs as `mlx` arrays.
#'
#' @param name Kernel name used in generated Metal code.
#' @param input_names Character vector naming the kernel inputs.
#' @param output_names Character vector naming the kernel outputs.
#' @param source Metal source for the kernel body. MLX generates the function
#'   signature automatically.
#' @param header Optional Metal source prepended before the generated function.
#' @param ensure_row_contiguous Logical. Should MLX make inputs row-contiguous
#'   before launching the kernel?
#' @param atomic_outputs Logical. Should output buffers use Metal atomic types?
#' @return A function that executes the compiled kernel and returns one `mlx`
#'   array for a single output or a named list of `mlx` arrays otherwise.
#' @seealso [mlx_compile()]
#' @seealso [mlx.core.fast.metal_kernel](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html)
#' @seealso [Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
#' @examples
#' \dontrun{
#' add_one <- mlx_metal_kernel(
#'   name = "add_one",
#'   input_names = "inp",
#'   output_names = "out",
#'   source = "
#'     uint elem = thread_position_in_grid.x;
#'     out[elem] = inp[elem] + (T)1;
#'   "
#' )
#'
#' x <- mlx_cast(as_mlx(1:8), "float32")
#' y <- add_one(
#'   inputs = list(x),
#'   output_shapes = list(c(length(x))),
#'   output_dtypes = "float32",
#'   grid = c(length(x), 1L, 1L),
#'   threadgroup = c(length(x), 1L, 1L),
#'   template = list(T = "float32")
#' )
#' }
#' @export
mlx_metal_kernel <- function(name,
                             input_names,
                             output_names,
                             source,
                             header = "",
                             ensure_row_contiguous = TRUE,
                             atomic_outputs = FALSE) {
  kernel_ptr <- cpp_mlx_metal_kernel_create(
    name = as.character(name),
    input_names = as.character(input_names),
    output_names = as.character(output_names),
    source = as.character(source),
    header = as.character(header),
    ensure_row_contiguous = isTRUE(ensure_row_contiguous),
    atomic_outputs = isTRUE(atomic_outputs)
  )

  function(inputs,
           output_shapes,
           output_dtypes,
           grid,
           threadgroup,
           template = list(),
           init_value = NULL,
           verbose = FALSE,
           device = NULL) {
    if (!is.list(inputs) || length(inputs) == 0) {
      stop("inputs must be a non-empty list.", call. = FALSE)
    }
    mlx_inputs <- lapply(inputs, as_mlx)

    if (!is.list(output_shapes)) {
      output_shapes <- list(output_shapes)
    }

    if (length(output_dtypes) == 1L && length(output_shapes) > 1L) {
      output_dtypes <- rep(output_dtypes, length(output_shapes))
    }

    if (is.null(device)) {
      device <- mlx_inputs[[1]]$device
    }

    result <- cpp_mlx_metal_kernel_call(
      kernel_xp = kernel_ptr,
      mlx_args = mlx_inputs,
      output_shapes = output_shapes,
      output_dtypes = as.character(output_dtypes),
      grid = as.integer(grid),
      threadgroup = as.integer(threadgroup),
      template_args = template,
      init_value = init_value,
      verbose = isTRUE(verbose),
      device_str = as.character(device)
    )

    if (length(result) == 1L) {
      result[[1]]
    } else {
      result
    }
  }
}
