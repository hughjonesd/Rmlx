#' Import an exported MLX function
#'
#' Loads a function previously exported with `mlx.export_function()` (Python)
#' and returns an R callable that accepts MLX-compatible arguments.
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
#' add_fn <- mlx_import_function(system.file("examples", "add.mlxfn", package = "Rmlx"))
#' x <- as_mlx(matrix(1:4, 2))
#' y <- as_mlx(matrix(5:8, 2))
#' add_fn(x, y)
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
