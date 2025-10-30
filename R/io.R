#' Save an MLX array to disk
#'
#' Persists an MLX tensor to a `.npy` file using MLX's native serialization.
#'
#' @param x Object coercible to `mlx`.
#' @param file Path to the output file. If the file does not end with `.npy`,
#'   the extension is appended automatically.
#' @return Invisibly returns the full path that was written, including the
#'   `.npy` suffix.
#' @seealso <https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.save>
#' @export
#' @examples
#' \dontrun{
#' path <- tempfile(fileext = ".mlx")
#' mlx_save(as_mlx(matrix(1:4, 2, 2), device = "cpu"), path)
#' restored <- mlx_load(path, device = "cpu")
#' }
mlx_save <- function(x, file) {
  x <- as_mlx(x)
  file <- .ensure_extension(path.expand(.validate_path(file)), ".npy")
  .ensure_parent_dir(file)

  cpp_mlx_save(x$ptr, file)
  invisible(file)
}

#' Load an MLX array from disk
#'
#' Restores a tensor saved with [mlx_save()] and optionally places it on a
#' specified device.
#'
#' @param file Path to a `.npy` file. The extension is appended automatically
#'   when missing.
#' @param device Target device for the loaded tensor (`"gpu"` or `"cpu"`).
#' @return An `mlx` array containing the file contents.
#' @seealso <https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.load>
#' @export
mlx_load <- function(file, device = mlx_default_device()) {
  file <- .ensure_extension(path.expand(.validate_path(file)), ".npy")
  if (!file.exists(file)) {
    stop("File '", file, "' does not exist.", call. = FALSE)
  }

  device <- match.arg(device, c("gpu", "cpu"))
  ptr <- cpp_mlx_load(file, device)
  .mlx_wrap_result(ptr, device)
}

#' Save MLX tensors to the safetensors format
#'
#' @param file Output path. `.safetensors` is appended automatically when omitted.
#' @param tensors Named list of objects coercible to `mlx`.
#' @param metadata Optional named character vector of metadata entries.
#' @return Invisibly returns the full path that was written.
#' @seealso <https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.save_safetensors>
#' @export
mlx_save_safetensors <- function(file, tensors, metadata = character()) {
  tensors <- .normalize_tensor_list(tensors)

  if (!is.null(metadata) && length(metadata)) {
    metadata_names <- names(metadata)
    if (is.null(metadata_names) || any(metadata_names == "")) {
      stop("`metadata` must be a named character vector.", call. = FALSE)
    }
    metadata <- setNames(as.character(metadata), metadata_names)
  } else {
    metadata <- character()
  }

  file <- .ensure_extension(path.expand(.validate_path(file)), ".safetensors")
  .ensure_parent_dir(file)

  tensor_ptrs <- lapply(tensors, `[[`, "ptr")
  tensor_names <- names(tensors)
  metadata_names <- names(metadata)
  if (is.null(metadata_names)) {
    metadata_names <- character()
  }
  metadata_values <- unname(metadata)

  cpp_mlx_save_safetensors(tensor_ptrs, tensor_names, metadata_names, metadata_values, file)
  invisible(file)
}

#' Load MLX tensors from the safetensors format
#'
#' @inheritParams mlx_load
#' @return A list containing:
#' \describe{
#'   \item{`tensors`}{Named list of `mlx` arrays.}
#'   \item{`metadata`}{Named character vector with the serialized metadata.}
#' }
#' @seealso <https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.load_safetensors>
#' @export
mlx_load_safetensors <- function(file, device = mlx_default_device()) {
  file <- .ensure_extension(path.expand(.validate_path(file)), ".safetensors")
  if (!file.exists(file)) {
    stop("File '", file, "' does not exist.", call. = FALSE)
  }

  device <- match.arg(device, c("gpu", "cpu"))
  cpp_mlx_load_safetensors(file, device)
}

#' Save MLX tensors to the GGUF format
#'
#' @param metadata Optional named list describing GGUF metadata. Values may be
#'   `NULL`, character vectors, or `mlx` arrays.
#' @inheritParams mlx_save_safetensors
#' @return Invisibly returns the full path that was written.
#' @seealso <https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.save_gguf>
#' @export
mlx_save_gguf <- function(file, tensors, metadata = list()) {
  tensors <- .normalize_tensor_list(tensors)
  metadata_payload <- .normalize_gguf_metadata(metadata)

  file <- .ensure_extension(path.expand(.validate_path(file)), ".gguf")
  .ensure_parent_dir(file)

  tensor_ptrs <- lapply(tensors, `[[`, "ptr")
  tensor_names <- names(tensors)
  metadata_names <- names(metadata_payload)
  if (is.null(metadata_names)) {
    metadata_names <- character()
  }

  cpp_mlx_save_gguf(tensor_ptrs, tensor_names, metadata_payload, metadata_names, file)
  invisible(file)
}

#' Load MLX tensors from the GGUF format
#'
#' @inheritParams mlx_load
#' @return A list containing:
#' \describe{
#'   \item{`tensors`}{Named list of `mlx` arrays.}
#'   \item{`metadata`}{Named list where values are `NULL`, character vectors, or
#'     `mlx` arrays depending on the GGUF entry type.}
#' }
#' @seealso <https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.load_gguf>
#' @export
mlx_load_gguf <- function(file, device = mlx_default_device()) {
  file <- .ensure_extension(path.expand(.validate_path(file)), ".gguf")
  if (!file.exists(file)) {
    stop("File '", file, "' does not exist.", call. = FALSE)
  }

  device <- match.arg(device, c("gpu", "cpu"))
  cpp_mlx_load_gguf(file, device)
}

.normalize_tensor_list <- function(tensors) {
  tensors <- as.list(tensors)
  if (!length(tensors)) {
    stop("`tensors` must contain at least one element.", call. = FALSE)
  }
  if (is.null(names(tensors)) || any(names(tensors) == "")) {
    stop("`tensors` must be a named list.", call. = FALSE)
  }
  lapply(tensors, as_mlx)
}

.normalize_gguf_metadata <- function(metadata) {
  metadata <- as.list(metadata)
  if (!length(metadata)) {
    return(setNames(list(), character()))
  }
  if (is.null(names(metadata)) || any(names(metadata) == "")) {
    stop("`metadata` must be a named list.", call. = FALSE)
  }

  out <- vector("list", length(metadata))
  names(out) <- names(metadata)
  for (nm in names(metadata)) {
    value <- metadata[[nm]]
    entry_type <- .gguf_entry_type(value)
    out[[nm]] <- switch(
      entry_type,
      array = list(type = "array", ptr = value$ptr),
      string = list(type = "string", value = value),
      string_vec = list(type = "string_vec", value = value)
    )
  }
  out
}

.gguf_entry_type <- function(value) {
  if (is.null(value)) {
    stop("GGUF metadata does not support NULL entries.", call. = FALSE)
  }
  if (inherits(value, "mlx")) {
    return("array")
  }
  if (is.character(value)) {
    if (!length(value)) {
      stop("Character metadata entries must have length >= 1.", call. = FALSE)
    }
    return(if (length(value) == 1L) "string" else "string_vec")
  }
  stop("Unsupported GGUF metadata entry of class: ", paste(class(value), collapse = ", "), call. = FALSE)
}

.ensure_extension <- function(path, ext) {
  if (.ends_with_ci(path, ext)) {
    path
  } else {
    paste0(path, ext)
  }
}
.ends_with_ci <- function(path, ext) {
  if (nchar(path) < nchar(ext)) {
    return(FALSE)
  }
  suffix <- substr(path, nchar(path) - nchar(ext) + 1L, nchar(path))
  tolower(suffix) == tolower(ext)
}

.ensure_parent_dir <- function(path) {
  dir_path <- dirname(path)
  if (!dir.exists(dir_path)) {
    stop("Directory '", dir_path, "' does not exist.", call. = FALSE)
  }
  invisible(NULL)
}

.validate_path <- function(path) {
  if (!is.character(path) || length(path) != 1L || is.na(path)) {
    stop("`file` must be a single, non-missing character string.", call. = FALSE)
  }
  path
}
