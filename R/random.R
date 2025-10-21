#' Sample from a uniform distribution on MLX tensors
#'
#' @param dim Integer vector giving the tensor shape.
#' @param min Lower bound for the uniform distribution.
#' @param max Upper bound for the uniform distribution.
#' @param dtype Desired MLX dtype ("float32" or "float64").
#' @param device Target device ("gpu" or "cpu").
#'
#' @return An `mlx` tensor whose entries are sampled uniformly.
#' @export
#'
mlx_rand_uniform <- function(dim, min = 0, max = 1,
                             dtype = c("float32", "float64"),
                             device = mlx_default_device()) {
  if (length(dim) == 0L) {
    stop("dim must contain at least one element.", call. = FALSE)
  }
  dim <- as.integer(dim)
  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_random_uniform(dim, min, max, dtype, device)
  new_mlx(ptr, dim, dtype, device)
}

#' Sample from a normal distribution on MLX tensors
#'
#' @inheritParams mlx_rand_uniform
#' @param mean Mean of the normal distribution.
#' @param sd Standard deviation of the normal distribution.
#' @return An `mlx` tensor with normally distributed entries.
#' @export
mlx_rand_normal <- function(dim, mean = 0, sd = 1,
                            dtype = c("float32", "float64"),
                            device = mlx_default_device()) {
  if (length(dim) == 0L) {
    stop("dim must contain at least one element.", call. = FALSE)
  }
  dim <- as.integer(dim)
  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_random_normal(dim, mean, sd, dtype, device)
  new_mlx(ptr, dim, dtype, device)
}

#' Sample Bernoulli random variables on MLX tensors
#'
#' @param dim Integer vector giving the tensor shape.
#' @param prob Probability of a one.
#' @param device Target device ("gpu" or "cpu").
#' @return An `mlx` boolean tensor.
#' @export
mlx_rand_bernoulli <- function(dim, prob = 0.5, device = mlx_default_device()) {
  if (length(dim) == 0L) {
    stop("dim must contain at least one element.", call. = FALSE)
  }
  if (prob < 0 || prob > 1) {
    stop("prob must be between 0 and 1.", call. = FALSE)
  }
  dim <- as.integer(dim)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_random_bernoulli(dim, prob, device)
  new_mlx(ptr, dim, "bool", device)
}
