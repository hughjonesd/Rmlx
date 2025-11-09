#' Import an exported MLX function
#'
#' Loads a function previously exported with the MLX Python utilities and
#' returns an R callable.
#'
#' Imported functions behave like regular R closures:
#' - Positional arguments are passed first and become the positional inputs
#'   the original MLX function expects.
#' - Named arguments (e.g. `bias = ...`) become MLX keyword arguments and must
#'   match the names that were used when exporting.
#' - Each argument is coerced to `mlx` via [as_mlx()] and automatically moved to
#'   the requested device/stream before execution.
#' - If the MLX function yields a single array the result is returned as an
#'   `mlx` object; multiple outputs are returned as a list in the order MLX
#'   produced them.
#'
#' Because `.mlxfn` files can bundle multiple traces (different shapes or
#' keyword combinations), the imported callable keeps a varargs (`...`)
#' signature. MLX selects the appropriate trace at runtime based on the shapes
#' and keyword names you provide.
#'
#' @param path Path to a `.mlxfn` file created via MLX export utilities.
#' @param device Default execution device (`"gpu"`, `"cpu"`, or an
#'   `mlx_stream`).
#' @return An R function. Calling it returns an `mlx` array if the imported
#'   function has a single output, or a list of `mlx` arrays otherwise.
#' @export
#' @examples
#' \
#' \dontrun{
#' add_fn <- mlx_import_function(
#'   system.file("testthat/fixtures/add_matrix.mlxfn", package = "Rmlx"),
#'   device = "cpu"
#' )
#' x <- as_mlx(matrix(1:4, 2, 2))
#' y <- as_mlx(matrix(5:8, 2, 2))
#' add_fn(x, bias = y)  # positional + keyword argument
#' }
mlx_import_function <- function(path, device = mlx_default_device()) {
  stopifnot(is.character(path), length(path) == 1L)
  normalized <- normalizePath(path, mustWork = TRUE)
  default_handle <- .mlx_resolve_device(device, mlx_default_device())
  ptr <- cpp_mlx_import_function(normalized)

  format_outputs <- function(result) {
    if (length(result) == 1L) {
      return(result[[1]])
    }
    result
  }

  function(..., .device = default_handle$device) {
    dots <- list(...)
    dot_names <- names(dots)
    if (is.null(dot_names)) {
      dot_names <- rep("", length(dots))
    }
    is_named <- !is.na(dot_names) & nzchar(dot_names)

    positional <- if (length(dots)) dots[!is_named] else list()
    kwargs <- if (length(dots)) dots[is_named] else list()

    handle <- .mlx_resolve_device(.device, default_handle$device)
    .mlx_eval_with_stream(handle, function(dev) {
      cast_to_device <- function(arg) {
        obj <- as_mlx(arg)
        if (obj$device == dev) {
          return(obj)
        }
        .mlx_cast(obj, device = dev)
      }

      positional_mlx <- lapply(positional, cast_to_device)
      kwargs_mlx <- lapply(kwargs, cast_to_device)

      args_ptrs <- lapply(positional_mlx, `[[`, "ptr")
      kwargs_ptrs <- lapply(kwargs_mlx, `[[`, "ptr")
      if (length(kwargs_ptrs)) {
        names(kwargs_ptrs) <- names(kwargs)
      }

      result <- cpp_mlx_call_imported(ptr, args_ptrs, kwargs_ptrs, dev)
      format_outputs(result)
    })
  }
}
