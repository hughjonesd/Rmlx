#' Save an MLX array to disk
#'
#' Persists an MLX array to a `.npy` file using MLX's native serialization.
#'
#' @param x Object coercible to `mlx`.
#' @param file Path to the output file. If the file does not end with `.npy`,
#'   the extension is appended automatically.
#' @return Invisibly returns the full path that was written, including the
#'   `.npy` suffix.
#' @seealso <https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.save>
#' @export
#' @examples
#' path <- tempfile(fileext = ".mlx")
#' mlx_save(as_mlx(matrix(1:4, 2, 2)), path)
#' restored <- mlx_load(path)
mlx_save <- function(x, file) {
  x <- as_mlx(x)
  file <- ensure_extension(path.expand(validate_path(file)), ".npy")
  ensure_parent_dir(file)

  cpp_mlx_save(x$ptr, file)
  invisible(file)
}

#' Load an MLX array from disk
#'
#' Restores an array saved with [mlx_save()].
#'
#' @param file Path to a `.npy` file. The extension is appended automatically
#'   when missing.
#' @return An `mlx` array containing the file contents.
#' @seealso <https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.load>
#' @export
mlx_load <- function(file) {
  file <- ensure_extension(path.expand(validate_path(file)), ".npy")
  if (!file.exists(file)) {
    stop("File '", file, "' does not exist.", call. = FALSE)
  }

  ptr <- cpp_mlx_load(file)
  new_mlx(ptr)
}

#' Save MLX arrays to the safetensors format
#'
#' @param file Output path. `.safetensors` is appended automatically when omitted.
#' @param arrays Named list of objects coercible to `mlx`.
#' @param metadata Optional named character vector of metadata entries.
#' @return Invisibly returns the full path that was written.
#' @seealso <https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.save_safetensors>
#' @export
mlx_save_safetensors <- function(file, arrays, metadata = character()) {
  arrays <- normalize_array_list(arrays)

  if (!is.null(metadata) && length(metadata)) {
    metadata_names <- names(metadata)
    if (is.null(metadata_names) || any(metadata_names == "")) {
      stop("`metadata` must be a named character vector.", call. = FALSE)
    }
    metadata <- stats::setNames(as.character(metadata), metadata_names)
  } else {
    metadata <- character()
  }

  file <- ensure_extension(path.expand(validate_path(file)), ".safetensors")
  ensure_parent_dir(file)

  array_ptrs <- lapply(arrays, `[[`, "ptr")
  array_names <- names(arrays)
  metadata_names <- names(metadata)
  if (is.null(metadata_names)) {
    metadata_names <- character()
  }
  metadata_values <- unname(metadata)

  cpp_mlx_save_safetensors(array_ptrs, array_names, metadata_names, metadata_values, file)
  invisible(file)
}

#' Load MLX arrays from the safetensors format
#'
#' @inheritParams mlx_load
#' @return A list containing:
#' \describe{
#'   \item{`tensors`}{Named list of `mlx` arrays.}
#'   \item{`metadata`}{Named character vector with the serialized metadata.}
#' }
#' @seealso <https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.load_safetensors>
#' @export
mlx_load_safetensors <- function(file) {
  file <- ensure_extension(path.expand(validate_path(file)), ".safetensors")
  if (!file.exists(file)) {
    stop("File '", file, "' does not exist.", call. = FALSE)
  }

  cpp_mlx_load_safetensors(file)
}

#' Save MLX arrays to the GGUF format
#'
#' @param metadata Optional named list describing GGUF metadata. Values may be
#'   character vectors or `mlx` arrays.
#' @inheritParams mlx_save_safetensors
#' @return Invisibly returns the full path that was written.
#' @seealso <https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.save_gguf>
#' @export
mlx_save_gguf <- function(file, arrays, metadata = list()) {
  arrays <- normalize_array_list(arrays)
  metadata_payload <- normalize_gguf_metadata(metadata)

  file <- ensure_extension(path.expand(validate_path(file)), ".gguf")
  ensure_parent_dir(file)

  array_ptrs <- lapply(arrays, `[[`, "ptr")
  array_names <- names(arrays)
  metadata_names <- names(metadata_payload)
  if (is.null(metadata_names)) {
    metadata_names <- character()
  }

  cpp_mlx_save_gguf(array_ptrs, array_names, metadata_payload, metadata_names, file)
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
mlx_load_gguf <- function(file) {
  file <- ensure_extension(path.expand(validate_path(file)), ".gguf")
  if (!file.exists(file)) {
    stop("File '", file, "' does not exist.", call. = FALSE)
  }

  cpp_mlx_load_gguf(file)
}

normalize_array_list <- function(arrays) {
  arrays <- as.list(arrays)
  if (!length(arrays)) {
    stop("`arrays` must contain at least one element.", call. = FALSE)
  }
  if (is.null(names(arrays)) || any(names(arrays) == "")) {
    stop("`arrays` must be a named list.", call. = FALSE)
  }
  lapply(arrays, as_mlx)
}

normalize_gguf_metadata <- function(metadata) {
  metadata <- as.list(metadata)
  if (!length(metadata)) {
    return(stats::setNames(list(), character()))
  }
  if (is.null(names(metadata)) || any(names(metadata) == "")) {
    stop("`metadata` must be a named list.", call. = FALSE)
  }

  out <- vector("list", length(metadata))
  names(out) <- names(metadata)
  for (nm in names(metadata)) {
    value <- metadata[[nm]]
    entry_type <- {
      if (is.null(value)) {
        stop("GGUF metadata does not support NULL entries.", call. = FALSE)
      }
      if (inherits(value, "mlx")) {
        "array"
      } else if (is.character(value)) {
        if (!length(value)) {
          stop("Character metadata entries must have length >= 1.", call. = FALSE)
        }
        if (length(value) == 1L) "string" else "string_vec"
      } else {
        stop(
          "Unsupported GGUF metadata entry of class: ",
          paste(class(value), collapse = ", "),
          call. = FALSE
        )
      }
    }
    out[[nm]] <- switch(
      entry_type,
      array = list(type = "array", ptr = value$ptr),
      string = list(type = "string", value = value),
      string_vec = list(type = "string_vec", value = value)
    )
  }
  out
}

ensure_extension <- function(path, ext) {
  has_ext <- nchar(path) >= nchar(ext) && {
    suffix <- substr(path, nchar(path) - nchar(ext) + 1L, nchar(path))
    tolower(suffix) == tolower(ext)
  }
  if (has_ext) path else paste0(path, ext)
}

ensure_parent_dir <- function(path) {
  dir_path <- dirname(path)
  if (!dir.exists(dir_path)) {
    stop("Directory '", dir_path, "' does not exist.", call. = FALSE)
  }
  invisible(NULL)
}

validate_path <- function(path) {
  if (!is.character(path) || length(path) != 1L || is.na(path)) {
    stop("`file` must be a single, non-missing character string.", call. = FALSE)
  }
  path
}
