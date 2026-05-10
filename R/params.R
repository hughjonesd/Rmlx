
#' @param device Execution target: supply `"gpu"`, `"cpu"`, or an
#'   `mlx_stream` created via [mlx_new_stream()]. By default, many
#'   functions use the [mlx_device()] of their first argument.
#' @param dtype Data type string. Supported types include:
#'   - Floating point: `"float32"`, `"float64"`
#'   - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`, `"uint16"`, `"uint32"`, `"uint64"`
#'   - Other: `"bool"`, `"complex64"`
#'
#'   `float64` arrays are CPU-only. Use `device = "cpu"` when creating or
#'   casting to `float64`, and cast back to `float32` before using the GPU.
#'   Not all functions support all types. See individual function documentation.
#' @param axis Single axis (1-indexed). Supply a positive integer between 1 and
#'   the array rank. Use `NULL` when the helper interprets it as "all axes" (see
#'   individual docs).
#' @param axes Integer vector of axes (1-indexed). Supply positive integers
#'   between 1 and the array rank. Many helpers interpret `NULL` to mean "all
#'   axes"—see the function details for specifics.
#' @param drop If `TRUE` (default), drop dimensions of length 1. If `FALSE`,
#'   retain all dimensions. Equivalent to `keepdims = TRUE` in underlying
#'   mlx functions.
#' @param dim Integer vector specifying array dimensions (shape).
#' @param x An mlx array, or an R array/matrix/vector that will be converted via [as_mlx()].
#'
#' @name common_params
#' @keywords internal
NULL

#' @param x An mlx array.
#'
#' @name mlx_array_required
#' @keywords internal
NULL

#' @param x An mlx matrix (2-dimensional array).
#'
#' @name mlx_matrix_required
#' @keywords internal
NULL

#' @param input Input mlx array. Shape depends on dimensionality (see individual functions).
#' @param weight Weight array. Shape depends on dimensionality (see individual functions).
#' @param stride Stride of the convolution. Can be a scalar or vector (length depends
#'   on dimensionality). Default: 1 for 1D, c(1,1) for 2D, c(1,1,1) for 3D.
#' @param padding Amount of zero padding. Can be a scalar or vector (length depends
#'   on dimensionality). Default: 0 for 1D, c(0,0) for 2D, c(0,0,0) for 3D.
#' @param dilation Spacing between kernel elements. Can be a scalar or vector (length
#'   depends on dimensionality). Default: 1 for 1D, c(1,1) for 2D, c(1,1,1) for 3D.
#' @param groups Number of blocked connections from input to output channels. Default: 1.
#'
#' @name conv_params
#' @keywords internal
NULL

#' @name ellipsis_ignored
#' @keywords internal
#' @param ... Additional arguments; ignored.
NULL

#' @name ellipsis_base
#' @keywords internal
#' @param ... Additional arguments forwarded to the corresponding base R implementation for signature compatibility.
NULL

#' CPU-only MLX operation note
#'
#' @details As of MLX 0.31.1, this operation only runs on CPU. Create or cast
#' the operands with `device = "cpu"` explicitly, or pass a `device = "cpu"`
#' argument. (Passing the argument won't affect the device of any
#' mlx object returned, just where this particular operation is run.)
#' @name mlx_cpu_only_operation
#' @keywords internal
NULL
