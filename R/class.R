#' Coerce R payload into the storage format expected by MLX
#'
#' @param x Input object (vector/array).
#' @param dtype Target MLX dtype.
#' @return Vector of numeric or complex values (or the original numeric input when
#'   it is already a double vector/matrix headed for float32/float64).
#' @noRd
coerce_payload <- function(x, dtype) {
  if (dtype %in% c("float32", "float64") &&
      is.double(x) &&
      is.atomic(x) &&
      !is.object(x)) {
    return(x)
  }

  switch(
    dtype,
    "bool" = {
      x_logical <- as.logical(x)
      if (anyNA(x_logical)) {
        stop("Logical NA values are not supported for MLX boolean arrays.", call. = FALSE)
      }
      as.numeric(x_logical)
    },
    "complex64" = as.complex(x),
    as.numeric(x)
  )
}

local_float64_cpu <- function(dtype, .local_envir = parent.frame()) {
  if (identical(dtype, "float64")) {
    local_device("cpu", .local_envir = .local_envir)
  }
}

#' Create MLX array from R object
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @inheritParams common_params
#' @param x Numeric, logical, or complex vector, matrix, or array to convert
#' @param dtype Data type for the MLX array. One of:
#'   - Floating point: `"float32"`, `"float64"`
#'   - Integer signed: `"int8"`, `"int16"`, `"int32"`, `"int64"`
#'   - Integer unsigned: `"uint8"`, `"uint16"`, `"uint32"`, `"uint64"`
#'   - Other: `"bool"`, `"complex64"`
#'
#'   If not specified, defaults to `"float32"` for numeric, `"bool"` for logical,
#'   and `"complex64"` for complex inputs.
#' @return An object of class `mlx`
#'
#' ## Integer types require explicit dtype
#'
#' R integer vectors (like `1:10`) convert to `float32` by default.
#' To create integer MLX arrays, you must explicitly specify `dtype`:
#'
#' ```r
#' x <- as_mlx(1:10, dtype = "int32")  # Creates int32 array
#' x <- as_mlx(1:10)                    # Creates float32 array
#' ```
#'
#' ## Type precision
#'
#' - `float64` is supported on CPU only. Use [with_device()] or [local_device()]
#'   to run float64 work on CPU.
#' - Integer arithmetic may promote types (e.g., int32 + int32 might → int64)
#' - Mixed integer/float operations promote to float
#'
#' ## Missing values
#'
#' MLX does not have an `NA` sentinel. When you pass numeric `NA` values from R,
#' they are stored as `NaN` inside MLX and returned to R as `NaN`.
#' Use [is.nan()] on MLX arrays if you need to detect them. [is.na()] on mlx
#' objects calls [is.nan()].
#'
#' ## Scalars
#'
#' MLX allows scalar values, with a zero-length dimension (`integer(0)`). These
#' are not usually what R users want. `as_mlx()` never returns a scalar; call
#' `[mlx_reshape(x, integer(0))][mlx_reshape()]` to create one explicitly, or
#' use `[mlx_array(..., allow_scalar = TRUE)][mlx_array()]`.
#'
#' @seealso [mlx.core.array](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.array)
#' @seealso [mlx-methods]
#' @export
#' @examples
#' # Default float32 for numeric
#' x <- as_mlx(c(1.5, 2.5, 3.5))
#' mlx_dtype(x)  # "float32"
#'
#' # R integers also default to float32
#' x <- as_mlx(1:10)
#' mlx_dtype(x)  # "float32"
#'
#' # Explicit integer types
#' x_int <- as_mlx(1:10, dtype = "int32")
#' mlx_dtype(x_int)  # "int32"
#'
#' # Unsigned integers
#' x_uint <- as_mlx(c(0, 128, 255), dtype = "uint8")
#'
#' # Logical → bool
#' mask <- as_mlx(c(TRUE, FALSE, TRUE))
#' mlx_dtype(mask)  # "bool"
as_mlx <- function(x, dtype = c("float32", "float64", "bool", "complex64",
                                 "int8", "int16", "int32", "int64",
                                 "uint8", "uint16", "uint32", "uint64")) {
  dtype_val <- if (missing(dtype)) {
    if (is.logical(x)) {
      "bool"
    } else if (is.complex(x)) {
      "complex64"
    } else {
      "float32"
    }
  } else {
    match.arg(dtype)
  }

  if (is_mlx(x)) {
    need_dtype <- !missing(dtype) && !identical(mlx_dtype(x), dtype_val)
    if (!need_dtype) return(x)

    ptr <- cpp_mlx_cast(x$ptr, dtype_val)
    return(new_mlx(ptr, dimnames = dimnames(x)))
  }

  is_supported <- (is.vector(x) && !is.list(x)) || is.matrix(x) || is.array(x)
  if (!is_supported) {
    stop("Cannot convert object of class ", class(x)[1], " to mlx")
  }

  dim_vec <- {
    dims <- dim(x)
    if (!is.null(dims)) {
      as.integer(dims)
    } else {
      as.integer(length(x))
    }
  }
  x_payload <- coerce_payload(x, dtype_val)

  local_float64_cpu(dtype_val)

  # Create MLX array via C++
  ptr <- cpp_mlx_from_r(x_payload, as.integer(dim_vec), dtype_val)

  # Create S3 object (dim is always read from MLX via dim.mlx())
  new_mlx(ptr, dimnames = .mlx_dimnames_from_r(x, dim_vec))
}

#' Force evaluation of an MLX operations
#'
#' By default MLX computations are lazy. `mlx_eval(x)` forces the computations
#' behind `x` to run. You can do the same by calling (e.g.)
#' [as.matrix(x)][as.matrix.mlx()].
#' @inheritParams mlx_array_required
#' @return The input object, invisibly.
#' @seealso [mlx.core.eval](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.eval)
#' @export
#' @examples
#' system.time(x <- mlx_rand_normal(1e7))
#' system.time(mlx_eval(x))
mlx_eval <- function(x) {
  stopifnot(is_mlx(x))
  cpp_mlx_eval(x$ptr)
  invisible(x)
}

#' Convert MLX array to R matrix
#'
#' MLX arrays with other than 2 dimensions are converted to
#' a 1 column matrix, with a warning.
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @inheritParams mlx_array_required
#' @inheritParams ellipsis_ignored
#' @return A vector, matrix or array (numeric or logical depending on dtype).
#' @export
#' @examples
#' x <- mlx_matrix(1:4, 2, 2)
#' as.matrix(x)
as.matrix.mlx <- function(x, ...) {
  x <- as.array.mlx(x, ...)
  if (length(dim(x)) != 2L) {
    warning("Converting array to 1-column matrix")
    dim(x) <- c(length(x), 1L)
  }

  x
}

#' Convert MLX array to base R objects
#'
#' `as_r()` mirrors base R coercion rules: MLX objects with `dim()` equal to
#' `NULL` return a plain vector, while higher-dimensional inputs return matrices
#' or arrays.
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @inheritParams mlx_array_required
#' @inheritParams ellipsis_ignored
#' @return A vector, matrix, or array depending on the dimensions of `x`.
#' @export
#' @seealso [as.array.mlx()], [as.vector.mlx()], [as.matrix.mlx()]
#' @examples
#' v <- as_mlx(1:3)
#' as_r(v)      # numeric vector
as_r <- function(x, ...) {
  stopifnot(is_mlx(x))
  local_float64_cpu(mlx_dtype(x))
  mlx_eval(x)
  out <- cpp_mlx_to_r(x$ptr)
  if (length(dim(x)) == 0L) {
    out <- as.vector(out)
    names(out) <- names(x)
    return(out)
  }

  # Be careful before changing the below; dim(), attributes() and
  # class can interact surprisingly.
  dim(out) <- dim(x)
  attrs <- attributes(x)
  attrs$names <- NULL
  attrs$class <- NULL
  attrs$mlx_dimnames <- NULL
  if (length(attrs)) {
    for (nm in names(attrs)) {
      attr(out, nm) <- attrs[[nm]]
    }
  }
  dimnames(out) <- dimnames(x)
  out
}

#' Convert MLX array to R array
#'
#' Always returns an R array using the MLX shape. One-dimensional MLX inputs
#' become 1-D arrays (with `dim` set to their length) instead of plain vectors.
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @inheritParams mlx_array_required
#' @inheritParams ellipsis_ignored
#' @return An R array with the same shape as the MLX input.
#' @export
#' @seealso [as_r()], [as.vector.mlx()], [as.matrix.mlx()]
#' @examples
#' x <- mlx_matrix(1:8, 2, 4)
#' as.array(x)
#'
#' v <- as_mlx(1:3)
#' as.array(v)  # 1-D array with dim 3
as.array.mlx <- function(x, ...) {
  stopifnot(is_mlx(x))
  local_float64_cpu(mlx_dtype(x))
  mlx_eval(x)
  out <- cpp_mlx_to_r(x$ptr)

  shape <- mlx_shape(x)
  if (length(shape) == 0L) {
    shape <- length(out)
  }

  dim(out) <- shape
  attrs <- attributes(x)
  attrs$names <- NULL
  attrs$class <- NULL
  attrs$mlx_dimnames <- NULL
  if (length(attrs)) {
    for (nm in names(attrs)) {
      attr(out, nm) <- attrs[[nm]]
    }
  }
  dimnames(out) <- dimnames(x)
  out
}

#' Convert MLX array to R vector
#'
#' Converts an MLX array to an R vector. Multi-dimensional arrays
#' are flattened in column-major order (R's default).
#'
#' @inherit mlx_float64_cpu_internal details
#'
#' @inheritParams mlx_array_required
#' @param mode Character string specifying the type of vector to return (passed to [base::as.vector()])
#' @inheritParams ellipsis_ignored
#' @return A vector of the specified mode.
#' @export
#' @examples
#' x <- as_mlx(-1:1)
#' as.vector(x)
#' as.logical(x)
#' as.numeric(x)
#'
#' # Multi-dimensional arrays are flattened
#' m <- mlx_matrix(1:6, 2, 3)
#' as.vector(m)  # Flattened in column-major order
as.vector.mlx <- function(x, mode = "any") {
  as.vector(as.array(x), mode = mode)
}

#' @export
#' @rdname as.vector.mlx
as.logical.mlx <- function(x, ...) {
  as.logical(as.vector(x))
}

#' @export
#' @rdname as.vector.mlx
as.double.mlx <- function(x, ...) {
  as.double(as.vector(x))
}

#' @rdname as.vector.mlx
#' @export
as.numeric.mlx <- as.double.mlx

#' @export
#' @rdname as.vector.mlx
as.integer.mlx <- function(x, ...) {
  as.integer(as.vector(x))
}


#' Test if object is an MLX array
#'
#' @param x Object to test
#' @return Logical scalar.
#' @export
#' @examples
#' x <- mlx_matrix(1:4, 2, 2)
#' is_mlx(x)
is_mlx <- function(x) {
  inherits(x, "mlx")
}

#' Internal helper to wrap pointer-returning C++ calls
#'
#' @param fn Function that takes `x$ptr` as its first argument and returns an
#'   external pointer.
#' @param x An mlx object providing the pointer.
#' @param ... Additional arguments forwarded to `fn` after `x$ptr`.
#' @return An mlx object wrapping the pointer returned by `fn`.
#' @noRd
.mlx_from_call <- function(fn, x, ...) {
  stopifnot(is_mlx(x))
  ptr <- fn(x$ptr, ...)
  new_mlx(ptr, dimnames = dimnames(x))
}

#' Internal constructor for mlx objects
#'
#' @param ptr External pointer to MLX array
#' @keywords internal
#' @noRd
new_mlx <- function(ptr, dimnames = NULL) {
  out <- structure(
    list(
      ptr = ptr
    ),
    class = "mlx"
  )
  if (!is.null(dimnames)) {
    dimnames(out) <- dimnames
  }
  out
}

#' Validate dimnames metadata for an mlx shape
#'
#' @param dimnames A list of character vectors or `NULL`.
#' @param shape Integer MLX shape.
#' @return Normalized dimnames list or `NULL`.
#' @noRd
.mlx_validate_dimnames <- function(dimnames, shape) {
  if (is.null(dimnames)) {
    return(NULL)
  }
  if (!is.list(dimnames)) {
    stop("dimnames must be a list.", call. = FALSE)
  }
  if (length(shape) == 0L) {
    stop("dimnames are not supported for MLX scalars.", call. = FALSE)
  }
  if (length(dimnames) != length(shape)) {
    stop("length of dimnames must match the number of dimensions.", call. = FALSE)
  }

  out <- vector("list", length(shape))
  for (i in seq_along(shape)) {
    dn <- dimnames[[i]]
    if (is.null(dn)) {
      out[i] <- list(NULL)
      next
    }
    if (length(dn) != shape[[i]]) {
      stop(
        sprintf(
          "length of dimnames[[%d]] (%d) must match dimension %d (%d).",
          i, length(dn), i, shape[[i]]
        ),
        call. = FALSE
      )
    }
    out[[i]] <- as.character(dn)
  }

  if (all(vapply(out, is.null, logical(1)))) {
    return(NULL)
  }
  names(out) <- names(dimnames)
  out
}

.mlx_dimnames_from_r <- function(x, shape) {
  dn <- dimnames(x)
  if (!is.null(dn)) {
    return(.mlx_validate_dimnames(dn, shape))
  }
  nm <- names(x)
  if (length(shape) == 1L && !is.null(nm)) {
    return(.mlx_validate_dimnames(list(nm), shape))
  }
  NULL
}

.mlx_dimnames_for_shape <- function(x, shape) {
  dn <- dimnames(x)
  if (is.null(dn) || !identical(as.integer(shape), as.integer(mlx_shape(x)))) {
    return(NULL)
  }
  dn
}

.mlx_dimnames_for_binary <- function(result, x, y) {
  shape <- mlx_shape(result)
  x_dn <- .mlx_dimnames_for_shape(x, shape)
  if (!is.null(x_dn)) {
    return(x_dn)
  }
  .mlx_dimnames_for_shape(y, shape)
}

.mlx_matrix_inverse_dimnames <- function(x) {
  dn <- dimnames(x)
  if (is.null(dn) || length(dn) != 2L) {
    return(NULL)
  }
  out <- list(dn[[2L]], dn[[1L]])
  if (all(vapply(out, is.null, logical(1)))) NULL else out
}

#' Dimnames and names for MLX arrays
#'
#' Get or set R-side dimname metadata on `mlx` arrays. Names are stored as
#' ordinary R metadata on the wrapper and are not written into MLX storage.
#'
#' @param x An object.
#' @param value Replacement names or dimnames.
#' @return The requested names, or `x` with updated metadata for replacement
#'   forms.
#'
#' `rownames()` and `colnames()` use these `dimnames()` methods through base R's
#' internal generic dispatch.
#'
#' @name mlx-dimnames
#' @aliases dimnames.mlx dimnames<-.mlx names.mlx names<-.mlx
#' @export
dimnames.mlx <- function(x) {
  attr(x, "mlx_dimnames", exact = TRUE)
}

#' @rdname mlx-dimnames
#' @export
`dimnames<-.mlx` <- function(x, value) {
  value <- .mlx_validate_dimnames(value, mlx_shape(x))
  attr(x, "mlx_dimnames") <- value
  x
}

#' @rdname mlx-dimnames
#' @export
names.mlx <- function(x) {
  dn <- dimnames(x)
  if (length(mlx_shape(x)) == 1L && !is.null(dn)) {
    return(dn[[1L]])
  }
  NULL
}

#' @rdname mlx-dimnames
#' @export
`names<-.mlx` <- function(x, value) {
  shape <- mlx_shape(x)
  if (length(shape) != 1L) {
    stop("names can only be set on one-dimensional mlx arrays.", call. = FALSE)
  }
  dimnames(x) <- if (is.null(value)) NULL else list(value)
  x
}
