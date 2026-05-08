#' CPU-only MLX operation note
#'
#' @details As of MLX 0.31.1, this operation only runs on CPU. Rmlx respects the
#'   input array's preferred device and does not silently move GPU-preferred
#'   operands to CPU. To use this operation, create or cast the operands with
#'   `device = "cpu"` explicitly.
#' @name mlx_cpu_only_operation
#' @keywords internal
NULL
