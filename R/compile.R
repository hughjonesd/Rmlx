#' Compile an MLX Function for Optimized Execution
#'
#' Returns a compiled version of a function that traces and optimizes the
#' computation graph on first call, then reuses the compiled graph for
#' subsequent calls with matching input shapes and types.
#'
#' @param f An R function that takes MLX arrays as arguments and returns
#'   MLX array(s). The function must be pure (no side effects) and use only
#'   MLX operations.
#' @param shapeless Logical. If `TRUE`, the compiled function won't recompile
#'   when input shapes change. However, changing input dtypes or number of
#'   dimensions still triggers recompilation. Default: `FALSE`
#'
#' @return A compiled function with the same signature as `f`. The first call
#'   will be slow (tracing and compilation), but subsequent calls will be
#'   much faster.
#'
#' @details
#' ## How Compilation Works
#'
#' When you call `mlx_compile(f)`, it returns a new function immediately without
#' any tracing. The actual compilation happens on the **first call** to the
#' compiled function:
#'
#' 1. **First call**: MLX traces the function with placeholder inputs, builds
#'    the computation graph, optimizes it (fusing operations, eliminating
#'    redundancy), and caches the result. This is slow.
#' 2. **Subsequent calls**: If inputs have the same shapes and dtypes, MLX
#'    reuses the cached compiled graph. This is fast.
#' 3. **Recompilation**: Occurs when input shapes change (unless `shapeless = TRUE`),
#'    input dtypes change, or the number of arguments changes.
#'
#' ## Requirements for Compiled Functions
#'
#' Your function must:
#' - Accept only MLX arrays as arguments
#' - Return MLX array(s) - either a single mlx object or a list of mlx objects
#' - Use only MLX operations (no conversion to R)
#' - Be pure (no side effects, no external state modification)
#'
#' Your function **cannot**:
#' - Print or evaluate arrays during execution (`print()`, `as.matrix()`,
#'   `as.numeric()`, `[[` extraction, etc.)
#' - Use control flow based on array values (`if (x > 0)` where `x` is an array)
#' - Modify external variables or have other side effects
#' - Return non-MLX values
#'
#' ## Performance Benefits
#'
#' - **Operation fusion**: Combines multiple operations into optimized kernels
#' - **Memory reduction**: Eliminates intermediate allocations
#' - **Overhead reduction**: Bypasses R/C++ call overhead for fused operations
#'
#' Typical speedups range from 2-10x for operation-heavy functions.
#'
#' ## Shapeless Compilation
#'
#' Setting `shapeless = TRUE` allows the compiled function to handle varying
#' input shapes without recompilation:
#'
#' ```r
#' # Regular compilation - recompiles for each new shape
#' fast_fn <- mlx_compile(matmul_fn)
#' fast_fn(mlx_zeros(c(10, 64)), weights)  # Compiles for shape (10, 64)
#' fast_fn(mlx_zeros(c(20, 64)), weights)  # Recompiles for shape (20, 64)
#'
#' # Shapeless compilation - compiles once
#' fast_fn <- mlx_compile(matmul_fn, shapeless = TRUE)
#' fast_fn(mlx_zeros(c(10, 64)), weights)  # Compiles once
#' fast_fn(mlx_zeros(c(20, 64)), weights)  # No recompilation!
#' ```
#'
#' Shapeless mode sacrifices some optimization opportunities
#' but avoids recompilation costs. Use it when processing variable-sized batches.
#'
#' @seealso [mlx_disable_compile()], [mlx_enable_compile()]
#' @seealso [mlx.core.compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
#'
#' @examples
#' # Simple example
#' matmul_add <- function(x, w, b) {
#'   (x %*% w) + b
#' }
#'
#' # Compile it (returns immediately, no tracing yet)
#' fast_fn <- mlx_compile(matmul_add)
#'
#' # First call: slow (traces and compiles)
#' x <- mlx_rand_normal(c(32, 128))
#' w <- mlx_rand_normal(c(128, 256))
#' b <- mlx_rand_normal(c(256))
#' result <- fast_fn(x, w, b)  # Compiles during this call
#'
#' # Subsequent calls: fast (uses cached graph)
#' batches <- replicate(10, mlx_rand_normal(c(32, 128)), simplify = FALSE)
#' for (bat in batches) {
#'   result <- fast_fn(bat, w, b)  # Uses cached graph
#' }
#'
#' # Multiple returns
#' forward_and_norm <- function(x, w) {
#'   y <- x %*% w
#'   norm <- sqrt(sum(y * y))
#'   list(y, norm)  # Return list of mlx objects
#' }
#'
#' compiled_fn <- mlx_compile(forward_and_norm)
#' results <- compiled_fn(x, w)  # Returns list(y, norm)
#'
#' @export
mlx_compile <- function(f, shapeless = FALSE) {
  if (!is.function(f)) {
    stop("First argument must be a function.", call. = FALSE)
  }

  # Create compiled function wrapper (fast - no tracing yet)
  compiled_ptr <- cpp_mlx_compile_create(f, isTRUE(shapeless))

  # Return R closure that calls the compiled function
  function(...) {
    args <- list(...)
    if (length(args) == 0) {
      stop("Compiled function requires at least one argument.", call. = FALSE)
    }

    # Convert arguments to mlx if needed
    mlx_args <- lapply(args, function(arg) {
      as_mlx(arg)
    })

    # Call compiled function
    # First call: traces and compiles (slow)
    # Subsequent calls: uses cached graph (fast)
    result <- cpp_mlx_compile_call(compiled_ptr, mlx_args)

    # Return single mlx or list of mlx
    if (length(result) == 1) {
      result[[1]]
    } else {
      result
    }
  }
}

#' Control Global Compilation Behavior
#'
#' These functions control whether MLX compilation is enabled globally.
#'
#' @description
#' - `mlx_disable_compile()` prevents all compilation globally. Compiled
#'   functions will execute without optimization.
#' - `mlx_enable_compile()` enables compilation (overrides the
#'   `MLX_DISABLE_COMPILE` environment variable).
#'
#' @details
#' These are useful for debugging (to check if compilation is causing issues)
#' or benchmarking (to measure compilation overhead vs speedup).
#'
#' You can also disable compilation by setting the `MLX_DISABLE_COMPILE`
#' environment variable before loading the package.
#'
#' @return Invisibly returns `NULL`.
#'
#' @examples
#' demo_fn <- mlx_compile(function(x) x + 1)
#' x <- mlx_rand_normal(c(4, 4))
#'
#' # Disable compilation for debugging
#' mlx_disable_compile()
#' demo_fn(x)  # Runs without optimization
#'
#' # Re-enable compilation
#' mlx_enable_compile()
#' demo_fn(x)  # Runs with optimization
#'
#' @rdname mlx_compile_control
#' @export
mlx_disable_compile <- function() {
  invisible(cpp_mlx_disable_compile())
}

#' @rdname mlx_compile_control
#' @export
mlx_enable_compile <- function() {
  invisible(cpp_mlx_enable_compile())
}
