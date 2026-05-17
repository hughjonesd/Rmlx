#' Drop all-empty dimnames metadata
#'
#' @param dn A dimnames list or `NULL`.
#' @return `NULL` when every dimname component is `NULL`; otherwise `dn`.
#' @noRd
dimnames_compact <- function(dn) {
  if (is.null(dn) || all(vapply(dn, is.null, logical(1)))) {
    return(NULL)
  }
  dn
}

#' Validate dimnames metadata for an mlx shape
#'
#' @param dn A list of character vectors or `NULL`.
#' @param shape Integer MLX shape.
#' @return Normalized dimnames list or `NULL`.
#' @noRd
dimnames_validate <- function(dn, shape) {
  if (is.null(dn)) {
    return(NULL)
  }
  if (!is.list(dn)) {
    stop("dimnames must be a list.", call. = FALSE)
  }
  if (length(shape) == 0L) {
    stop("dimnames are not supported for MLX scalars.", call. = FALSE)
  }
  if (length(dn) != length(shape)) {
    stop("length of dimnames must match the number of dimensions.", call. = FALSE)
  }

  out <- vector("list", length(shape))
  for (i in seq_along(shape)) {
    dim_names <- dn[[i]]
    if (is.null(dim_names)) {
      out[i] <- list(NULL)
      next
    }
    if (length(dim_names) != shape[[i]]) {
      stop(
        sprintf(
          "length of dimnames[[%d]] (%d) must match dimension %d (%d).",
          i, length(dim_names), i, shape[[i]]
        ),
        call. = FALSE
      )
    }
    out[[i]] <- as.character(dim_names)
  }

  names(out) <- names(dn)
  dimnames_compact(out)
}

#' Infer dimnames from an R object
#'
#' @param x R vector, matrix, or array converted to mlx.
#' @param shape Integer MLX shape for the converted object.
#' @return Normalized dimnames list or `NULL`.
#' @noRd
dimnames_from_r <- function(x, shape) {
  dn <- dimnames(x)
  if (!is.null(dn)) {
    return(dimnames_validate(dn, shape))
  }
  nm <- names(x)
  if (length(shape) == 1L && !is.null(nm)) {
    return(dimnames_validate(list(nm), shape))
  }
  NULL
}

#' Return dimnames only when shape matches
#'
#' @param x An mlx object.
#' @param shape Candidate output shape.
#' @return `x` dimnames when `shape` equals `dim(x)`; otherwise `NULL`.
#' @noRd
dimnames_if_shape_matches <- function(x, shape) {
  dn <- dimnames(x)
  if (is.null(dn) || !identical(as.integer(shape), as.integer(mlx_shape(x)))) {
    return(NULL)
  }
  dn
}

#' Choose dimnames for binary-operation results
#'
#' @param result Binary operation result.
#' @param x,y Binary operands.
#' @return Dimnames from the first same-shaped operand with names, or `NULL`.
#' @noRd
dimnames_from_binary_operands <- function(result, x, y) {
  shape <- mlx_shape(result)
  x_dn <- dimnames_if_shape_matches(x, shape)
  if (!is.null(x_dn)) {
    return(x_dn)
  }
  dimnames_if_shape_matches(y, shape)
}

#' Combine dimnames when binding arrays
#'
#' @param objs List of mlx arrays being bound.
#' @param axis One-indexed axis along which arrays are concatenated.
#' @return Combined dimnames for the bound result.
#' @noRd
dimnames_bind <- function(objs, axis) {
  ref_shape <- mlx_shape(objs[[1L]])
  out <- vector("list", length(ref_shape))

  for (ax in seq_along(ref_shape)) {
    dns <- lapply(objs, function(obj) {
      dn <- dimnames(obj)
      if (is.null(dn)) NULL else dn[[ax]]
    })

    if (ax == axis) {
      if (all(vapply(dns, is.null, logical(1)))) {
        out[ax] <- list(NULL)
      } else {
        pieces <- Map(function(nm, obj) {
          nm %||% rep.int(NA_character_, mlx_shape(obj)[[ax]])
        }, dns, objs)
        out[[ax]] <- unlist(pieces, use.names = FALSE)
      }
      next
    }

    first <- dns[[1L]]
    if (!is.null(first) && all(vapply(dns, identical, logical(1), y = first))) {
      out[[ax]] <- first
    } else {
      out[ax] <- list(NULL)
    }
  }

  dimnames_compact(out)
}

#' Swap matrix dimnames for inverse-like outputs
#'
#' @param x An mlx matrix.
#' @return `list(colnames(x), rownames(x))`, compacted to `NULL` if unnamed.
#' @noRd
dimnames_matrix_inverse <- function(x) {
  dn <- dimnames(x)
  if (is.null(dn) || length(dn) != 2L) {
    return(NULL)
  }
  dimnames_compact(list(dn[[2L]], dn[[1L]]))
}

#' Transform dimnames after reducing axes
#'
#' @param x An mlx array.
#' @param axes One-indexed axes reduced by the operation.
#' @param drop Whether reduced axes are dropped from the result.
#' @return Result dimnames after dropping or nulling reduced axes.
#' @noRd
dimnames_reduction <- function(x, axes, drop) {
  dn <- dimnames(x)
  if (is.null(dn) || is.null(axes) || !length(axes)) {
    return(NULL)
  }
  axes <- sort(unique(as.integer(axes)))
  if (isTRUE(drop)) {
    dn <- dn[-axes]
  } else {
    dn[axes] <- rep(list(NULL), length(axes))
  }
  dimnames_compact(dn)
}

#' Choose dimnames for linear-system solve results
#'
#' @param a Coefficient matrix.
#' @param b Right-hand side vector or matrix.
#' @return `solve()`-style dimnames for the solution.
#' @noRd
dimnames_solve <- function(a, b) {
  if (length(dim(b)) < 2L) {
    return(dimnames_compact(list(colnames(a))))
  }
  dimnames_compact(list(colnames(a), colnames(b)))
}

#' Choose dimnames for diagonal extraction
#'
#' @param x Source array.
#' @param result_shape Integer shape of the diagonal result.
#' @param offset Diagonal offset.
#' @param axis1,axis2 One-indexed axes defining diagonal planes.
#' @return Dimnames for `diag()`/`mlx_diagonal()` extraction.
#' @noRd
dimnames_diagonal <- function(x, result_shape, offset, axis1, axis2) {
  dn <- dimnames(x)
  if (is.null(dn) || length(mlx_shape(x)) < 2L || !length(result_shape)) {
    return(NULL)
  }

  axes <- c(axis1, axis2)
  remaining <- setdiff(seq_along(dn), axes)
  out <- dn[remaining]

  diag_len <- result_shape[[length(result_shape)]]
  diag_names <- NULL
  axis1_names <- dn[[axis1]]
  axis2_names <- dn[[axis2]]
  if (!is.null(axis1_names) && !is.null(axis2_names) && diag_len > 0L) {
    if (offset >= 0L) {
      axis1_pos <- seq_len(diag_len)
      axis2_pos <- axis1_pos + offset
    } else {
      axis2_pos <- seq_len(diag_len)
      axis1_pos <- axis2_pos - offset
    }
    candidate1 <- axis1_names[axis1_pos]
    candidate2 <- axis2_names[axis2_pos]
    if (identical(candidate1, candidate2)) {
      diag_names <- candidate1
    }
  }

  dimnames_compact(c(out, list(diag_names)))
}

#' Permute dimnames
#'
#' @param x Source array.
#' @param perm One-indexed axis permutation.
#' @return Permuted dimnames, or `NULL` when `x` is unnamed.
#' @noRd
dimnames_permute <- function(x, perm) {
  dn <- dimnames(x)
  if (is.null(dn)) {
    return(NULL)
  }
  dimnames_compact(dn[perm])
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
  value <- dimnames_validate(value, mlx_shape(x))
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
