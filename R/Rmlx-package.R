#' Rmlx: R Interface to Apple's MLX Arrays
#'
#' This package provides an R interface to Apple's MLX (Machine Learning eXchange)
#' library for GPU-accelerated array operations on Apple Silicon.
#'
#' @section Key Features:
#' \itemize{
#'   \item Lazy evaluation: Operations are not computed until explicitly evaluated
#'   \item GPU acceleration: Leverage Metal on Apple Silicon
#'   \item Familiar syntax: S3 methods for standard R operations
#'   \item Unified memory: Efficient data sharing between CPU and GPU
#' }
#'
#' @section Main Functions:
#' \itemize{
#'   \item \code{\link{as_mlx}}: Convert R objects to MLX arrays
#'   \item \code{\link{as.matrix.mlx}}: Convert MLX arrays back to R
#'   \item \code{\link{mlx_eval}}: Force evaluation of lazy operations
#'   \item Arithmetic: \code{+}, \code{-}, \code{*}, \code{/}, \code{^}
#'   \item Matrix ops: \code{\%*\%}, \code{t}, \code{crossprod}, \code{tcrossprod}
#'   \item Reductions: \code{sum}, \code{mean}, \code{colMeans}, \code{rowMeans}
#' }
#'
#' @section Lazy Evaluation:
#' MLX arrays use lazy evaluation by default. Operations are recorded but not
#' executed until:
#' \itemize{
#'   \item You call \code{mlx_eval(x)}
#'   \item You convert to R with \code{as.matrix(x)}
#'   \item The result is needed for another computation
#' }
#'
#' @section Device Management:
#' Use \code{\link{mlx_default_device}} to control whether arrays are created
#' on GPU (default) or CPU.
#'
#' @docType package
#' @name Rmlx-package
#' @aliases Rmlx
#' @useDynLib Rmlx, .registration = TRUE
#' @importFrom Rcpp sourceCpp
"_PACKAGE"
