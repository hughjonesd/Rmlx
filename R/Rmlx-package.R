#' Rmlx: R Interface to Apple's MLX Arrays
#'
#' This package provides an R interface to Apple's [MLX](https://mlx-framework.org/)
#' (Machine Learning eXchange) library for GPU-accelerated array operations on
#' Apple Silicon.
#'
#' # Key Features
#' * Lazy evaluation: Operations are not computed until explicitly evaluated
#' * GPU acceleration: Leverage Metal on Apple Silicon
#' * Familiar syntax: S3 methods for standard R operations
#' * Unified memory: Efficient data sharing between CPU and GPU
#'
#' # Lazy Evaluation
#' MLX arrays use lazy evaluation by default. Operations are recorded but not
#' executed until:
#' * You call [mlx_eval()]
#' * You convert to R with [as.matrix()] or [as.vector()]
#' * The result is needed for another computation
#'
#' The package implements most of the C++ API via calls with the `mlx_` prefix,
#' but it also ships [S3 methods for many base generics][mlx-methods],
#' so common R matrix operations continue to work on MLX arrays. R conventions
#' are used throughout: for example, indexing is 1-based.
#'
#'
#'
#' @docType package
#' @name Rmlx-package
#' @aliases Rmlx
#' @useDynLib Rmlx, .registration = TRUE
#' @importFrom Rcpp sourceCpp
"_PACKAGE"
