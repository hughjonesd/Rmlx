#' Rmlx: R Interface to Apple's MLX Arrays
#'
#' This package provides an R interface to Apple's MLX (Machine Learning eXchange)
#' library for GPU-accelerated array operations on Apple Silicon.
#'
#' # Key Features
#' * Lazy evaluation: Operations are not computed until explicitly evaluated
#' * GPU acceleration: Leverage Metal on Apple Silicon
#' * Familiar syntax: S3 methods for standard R operations
#' * Unified memory: Efficient data sharing between CPU and GPU
#' 
#' # Main Functions
#' * [as_mlx()]: Convert R objects to MLX arrays
#' * [as.matrix.mlx()]: Convert MLX arrays back to R
#' * [mlx_eval()]: Force evaluation of lazy operations
#' * Arithmetic: `+`, `-`, `*`, `/`, `^`
#' * Matrix ops: `%*%`, `t`, `crossprod`, `tcrossprod`
#' * Reductions: `sum`, `mean`, `colMeans`, `rowMeans`
#' 
#' # Lazy Evaluation
#' MLX arrays use lazy evaluation by default. Operations are recorded but not
#' executed until:
#' * You call `mlx_eval(x)`
#' * You convert to R with `as.matrix(x)`
#' * The result is needed for another computation
#' 
#' # Device Management
#' Use [mlx_default_device()] to control whether arrays are created
#' on GPU (default) or CPU. All mlx arrays are stored in `float32`
#' regardless of device. Use base R arrays if you require `float64` math.
#'
#' @docType package
#' @name Rmlx-package
#' @aliases Rmlx
#' @useDynLib Rmlx, .registration = TRUE
#' @importFrom Rcpp sourceCpp
"_PACKAGE"
