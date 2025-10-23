#' Sample from a uniform distribution on MLX tensors
#'
#' @inheritParams mlx_creation_params
#' @param dtype Desired MLX dtype ("float32" or "float64").
#' @param min Lower bound for the uniform distribution.
#' @param max Upper bound for the uniform distribution.
#'
#' @return An `mlx` tensor whose entries are sampled uniformly.
#' @export
#' @examples
#' noise <- mlx_rand_uniform(c(2, 2), min = -1, max = 1)
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
#' @inheritParams mlx_creation_params
#' @param dtype Desired MLX dtype ("float32" or "float64").
#' @param mean Mean of the normal distribution.
#' @param sd Standard deviation of the normal distribution.
#' @return An `mlx` tensor with normally distributed entries.
#' @export
#' @examples
#' weights <- mlx_rand_normal(c(3, 3), mean = 0, sd = 0.1)
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
#' @inheritParams mlx_creation_params
#' @param dtype Desired MLX dtype ("float32" or "float64").
#' @param prob Probability of a one.
#' @return An `mlx` boolean tensor.
#' @export
#' @examples
#' mask <- mlx_rand_bernoulli(c(4, 4), prob = 0.3)
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

#' Sample from the Gumbel distribution on MLX tensors
#'
#' @inheritParams mlx_creation_params
#' @param dtype Desired MLX dtype ("float32" or "float64").
#' @return An `mlx` tensor with Gumbel-distributed entries.
#' @export
#' @examples
#' samples <- mlx_rand_gumbel(c(2, 3))
mlx_rand_gumbel <- function(dim, dtype = c("float32", "float64"),
                            device = mlx_default_device()) {
  if (length(dim) == 0L) {
    stop("dim must contain at least one element.", call. = FALSE)
  }
  dim <- as.integer(dim)
  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_random_gumbel(dim, dtype, device)
  new_mlx(ptr, dim, dtype, device)
}

#' Sample from a truncated normal distribution on MLX tensors
#'
#' @inheritParams mlx_creation_params
#' @param dtype Desired MLX dtype ("float32" or "float64").
#' @param lower Lower bound of the truncated normal.
#' @param upper Upper bound of the truncated normal.
#' @return An `mlx` tensor with truncated normally distributed entries.
#' @export
#' @examples
#' samples <- mlx_rand_truncated_normal(-1, 1, c(5, 5))
mlx_rand_truncated_normal <- function(lower, upper, dim,
                                      dtype = c("float32", "float64"),
                                      device = mlx_default_device()) {
  if (length(dim) == 0L) {
    stop("dim must contain at least one element.", call. = FALSE)
  }
  if (!is.numeric(lower) || length(lower) != 1) {
    stop("lower must be a single numeric value.", call. = FALSE)
  }
  if (!is.numeric(upper) || length(upper) != 1) {
    stop("upper must be a single numeric value.", call. = FALSE)
  }
  dim <- as.integer(dim)
  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_random_truncated_normal(lower, upper, dim, dtype, device)
  new_mlx(ptr, dim, dtype, device)
}

#' Sample from a multivariate normal distribution on MLX tensors
#'
#' @inheritParams mlx_creation_params
#' @param dtype Desired MLX dtype ("float32" or "float64").
#' @param mean An `mlx` tensor or vector for the mean.
#' @param cov An `mlx` tensor or matrix for the covariance.
#' @param device Target device ("cpu" only). Note: this function requires CPU
#'   due to SVD decomposition of the covariance matrix; GPU device is not currently
#'   supported.
#' @return An `mlx` tensor with samples from the multivariate normal.
#' @export
#' @examples
#' mean <- as_mlx(c(0, 0), device = "cpu")
#' cov <- as_mlx(matrix(c(1, 0, 0, 1), 2, 2), device = "cpu")
#' samples <- mlx_rand_multivariate_normal(c(100, 2), mean, cov, device = "cpu")
mlx_rand_multivariate_normal <- function(dim, mean, cov,
                                         dtype = c("float32", "float64"),
                                         device = "cpu") {
  if (length(dim) == 0L) {
    stop("dim must contain at least one element.", call. = FALSE)
  }

  # Convert mean and cov to mlx if needed
  if (!is.mlx(mean)) {
    mean <- as_mlx(mean, device = device)
  }
  if (!is.mlx(cov)) {
    cov <- as_mlx(cov, device = device)
  }

  dim <- as.integer(dim)
  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_random_multivariate_normal(mean, cov, dim, dtype, device)
  # Get the actual output shape from the result
  output_shape <- cpp_mlx_shape(ptr)
  new_mlx(ptr, output_shape, dtype, device)
}
