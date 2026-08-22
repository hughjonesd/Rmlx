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
#' - Each argument is coerced to `mlx` via [as_mlx()].
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
#' @return An R function. Calling it returns an `mlx` array if the imported
#'   function has a single output, or a list of `mlx` arrays otherwise.
#' @export
#' @examplesIf mlx_has_gpu()
#' fixture_names <- c("add_matrix.mlxfn", "add_matrix_pre_metadata.mlxfn")
#' fixture_paths <- system.file("extdata", fixture_names, package = "Rmlx")
#' add_fn <- NULL
#' for (fixture_path in fixture_paths) {
#'   add_fn <- tryCatch(
#'     mlx_import_function(fixture_path),
#'     error = function(err) NULL
#'   )
#'   if (!is.null(add_fn)) break
#' }
#' stopifnot(!is.null(add_fn))
#' x <- mlx_matrix(1:4, 2, 2)
#' y <- mlx_matrix(5:8, 2, 2)
#' add_fn(x, y)
mlx_import_function <- function(path) {
  stopifnot(is.character(path), length(path) == 1L)
  normalized <- normalizePath(path, mustWork = TRUE)
  ptr <- cpp_mlx_import_function(normalized)

  format_outputs <- function(result) {
    if (length(result) == 1L) {
      return(result[[1]])
    }
    result
  }

  function(...) {
    dots <- list(...)
    dot_names <- names(dots)
    if (is.null(dot_names)) {
      dot_names <- rep("", length(dots))
    }
    is_named <- !is.na(dot_names) & nzchar(dot_names)

    positional <- if (length(dots)) dots[!is_named] else list()
    kwargs <- if (length(dots)) dots[is_named] else list()

    positional_mlx <- lapply(positional, as_mlx)
    kwargs_mlx <- lapply(kwargs, as_mlx)

    args_ptrs <- lapply(positional_mlx, `[[`, "ptr")
    kwargs_ptrs <- lapply(kwargs_mlx, `[[`, "ptr")
    if (length(kwargs_ptrs)) {
      names(kwargs_ptrs) <- names(kwargs)
    }

    result <- cpp_mlx_call_imported(ptr, args_ptrs, kwargs_ptrs)
    format_outputs(result)
  }
}
