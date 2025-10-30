#' Common Parameter Documentation
#'
#' @param device Device for computation: `"gpu"` or `"cpu"`. Default: `mlx_default_device()`.
#' @param dtype Data type string. Supported types include:
#'   - Floating point: `"float32"`, `"float64"`
#'   - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`, `"uint16"`, `"uint32"`, `"uint64"`
#'   - Other: `"bool"`, `"complex64"`
#'
#'   Not all functions support all types. See individual function documentation.
#' @param axis Axis or axes to operate on (1-indexed). Negative values count from
#'   the end. `NULL` operates on all axes or the entire array.
#' @param drop If `TRUE` (default), drop dimensions of length 1. If `FALSE`,
#'   retain all dimensions. Equivalent to `keepdims = TRUE` in underlying
#'   mlx functions.
#' @param dim Integer vector specifying array dimensions (shape).
#' @param x An mlx array, or an R array/matrix/vector that will be converted via [as_mlx()].
#'
#' @name common_params
#' @keywords internal
NULL


#' Parameters for Functions Requiring MLX Arrays
#'
#' @param x An mlx array.
#'
#' @name mlx_array_required
#' @keywords internal
NULL


#' Parameters for Functions Requiring MLX Matrices
#'
#' @param x An mlx matrix (2-dimensional array).
#'
#' @name mlx_matrix_required
#' @keywords internal
NULL


#' Common Convolution Parameters
#'
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

