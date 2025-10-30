#' Base R generics with mlx support
#'
#' Rmlx provides methods for a selection of base R generics so that existing
#' code continues to work after converting objects with [as_mlx()]. This helper
#' returns the list of supported generics in alphabetical order.
#'
#' @return Character vector of generic/function names that have mlx methods.
#' @export
#' @examples
#' mlx_base_methods()
mlx_base_methods <- function() {
  methods <- c(
    "%*%",
    "[",
    "[<-",
    "Ops",
    "Math",
    "Summary",
    "aperm",
    "as.array",
    "as.matrix",
    "as.vector",
    "cbind",
    "diag",
    "dim",
    "dim<-",
    "fft",
    "kronecker",
    "length",
    "outer",
    "print",
    "rbind",
    "str"
  )
  sort(unique(methods))
}
