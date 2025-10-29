#' Sample from a uniform distribution on mlx arrays
#'
#' @inheritParams mlx_creation_params
#' @param dtype Desired MLX dtype ("float32" or "float64").
#' @param min Lower bound for the uniform distribution.
#' @param max Upper bound for the uniform distribution.
#'
#' @return An mlx array whose entries are sampled uniformly.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.uniform}
#' @export
#' @examples
#' noise <- mlx_rand_uniform(c(2, 2), min = -1, max = 1)
mlx_rand_uniform <- function(dim, min = 0, max = 1,
                             dtype = c("float32", "float64"),
                             device = mlx_default_device()) {
  dim <- .validate_shape(dim)
  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_random_uniform(dim, min, max, dtype, device)
  new_mlx(ptr, dim, dtype, device)
}

#' Sample from a normal distribution on mlx arrays
#'
#' @inheritParams mlx_creation_params
#' @param dtype Desired MLX dtype ("float32" or "float64").
#' @param mean Mean of the normal distribution.
#' @param sd Standard deviation of the normal distribution.
#' @return An mlx array with normally distributed entries.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.normal}
#' @export
#' @examples
#' weights <- mlx_rand_normal(c(3, 3), mean = 0, sd = 0.1)
mlx_rand_normal <- function(dim, mean = 0, sd = 1,
                            dtype = c("float32", "float64"),
                            device = mlx_default_device()) {
  dim <- .validate_shape(dim)
  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_random_normal(dim, mean, sd, dtype, device)
  new_mlx(ptr, dim, dtype, device)
}

#' Sample Bernoulli random variables on mlx arrays
#'
#' @inheritParams mlx_creation_params
#' @param prob Probability of a one.
#' @return An mlx boolean array.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.bernoulli}
#' @export
#' @examples
#' mask <- mlx_rand_bernoulli(c(4, 4), prob = 0.3)
mlx_rand_bernoulli <- function(dim, prob = 0.5, device = mlx_default_device()) {
  dim <- .validate_shape(dim)
  if (prob < 0 || prob > 1) {
    stop("prob must be between 0 and 1.", call. = FALSE)
  }
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_random_bernoulli(dim, prob, device)
  new_mlx(ptr, dim, "bool", device)
}

#' Sample from the Gumbel distribution on mlx arrays
#'
#' @inheritParams mlx_creation_params
#' @param dtype Desired MLX dtype ("float32" or "float64").
#' @return An mlx array with Gumbel-distributed entries.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.gumbel}
#' @export
#' @examples
#' samples <- mlx_rand_gumbel(c(2, 3))
mlx_rand_gumbel <- function(dim, dtype = c("float32", "float64"),
                            device = mlx_default_device()) {
  dim <- .validate_shape(dim)
  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_random_gumbel(dim, dtype, device)
  new_mlx(ptr, dim, dtype, device)
}

#' Sample from a truncated normal distribution on mlx arrays
#'
#' @inheritParams mlx_creation_params
#' @param dtype Desired MLX dtype ("float32" or "float64").
#' @param lower Lower bound of the truncated normal.
#' @param upper Upper bound of the truncated normal.
#' @return An mlx array with truncated normally distributed entries.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.truncated_normal}
#' @export
#' @examples
#' samples <- mlx_rand_truncated_normal(-1, 1, c(5, 5))
mlx_rand_truncated_normal <- function(lower, upper, dim,
                                      dtype = c("float32", "float64"),
                                      device = mlx_default_device()) {
  dim <- .validate_shape(dim)
  if (!is.numeric(lower) || length(lower) != 1) {
    stop("lower must be a single numeric value.", call. = FALSE)
  }
  if (!is.numeric(upper) || length(upper) != 1) {
    stop("upper must be a single numeric value.", call. = FALSE)
  }
  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_random_truncated_normal(lower, upper, dim, dtype, device)
  new_mlx(ptr, dim, dtype, device)
}

#' Sample from a multivariate normal distribution on mlx arrays
#'
#' @inheritParams mlx_creation_params
#' @param dtype Desired MLX dtype ("float32" or "float64").
#' @param mean An mlx array or vector for the mean.
#' @param cov An mlx array or matrix for the covariance.
#' @param device Target device ("cpu" only). Note: this function requires CPU
#'   due to SVD decomposition of the covariance matrix; GPU device is not currently
#'   supported.
#' @return An mlx array with samples from the multivariate normal.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.multivariate_normal}
#' @export
#' @examples
#' mean <- as_mlx(c(0, 0), device = "cpu")
#' cov <- as_mlx(matrix(c(1, 0, 0, 1), 2, 2), device = "cpu")
#' samples <- mlx_rand_multivariate_normal(c(100, 2), mean, cov, device = "cpu")
mlx_rand_multivariate_normal <- function(dim, mean, cov,
                                         dtype = c("float32", "float64"),
                                         device = "cpu") {
  dim <- .validate_shape(dim)

  # Convert mean and cov to mlx if needed
  if (!is.mlx(mean)) {
    mean <- as_mlx(mean, device = device)
  }
  if (!is.mlx(cov)) {
    cov <- as_mlx(cov, device = device)
  }

  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_random_multivariate_normal(mean, cov, dim, dtype, device)
  # Get the actual output shape from the result
  output_shape <- cpp_mlx_shape(ptr)
  new_mlx(ptr, output_shape, dtype, device)
}

#' Sample from the Laplace distribution on mlx arrays
#'
#' @inheritParams mlx_creation_params
#' @param dtype Desired MLX dtype ("float32" or "float64").
#' @param loc Location parameter (mean) of the Laplace distribution.
#' @param scale Scale parameter (diversity) of the Laplace distribution.
#' @return An mlx array with Laplace-distributed entries.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.laplace}
#' @export
#' @examples
#' samples <- mlx_rand_laplace(c(2, 3), loc = 0, scale = 1)
mlx_rand_laplace <- function(dim, loc = 0, scale = 1,
                             dtype = c("float32", "float64"),
                             device = mlx_default_device()) {
  dim <- .validate_shape(dim)
  if (!is.numeric(loc) || length(loc) != 1) {
    stop("loc must be a single numeric value.", call. = FALSE)
  }
  if (!is.numeric(scale) || length(scale) != 1 || scale <= 0) {
    stop("scale must be a single positive numeric value.", call. = FALSE)
  }
  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_random_laplace(dim, loc, scale, dtype, device)
  new_mlx(ptr, dim, dtype, device)
}

#' Sample from a categorical distribution on mlx arrays
#'
#' Samples indices from categorical distributions. Each row (or slice along the
#' specified axis) represents a separate categorical distribution over classes.
#'
#' @param logits A matrix or mlx array of log-probabilities. The values don't
#'   need to be normalized (the function applies softmax internally). For a single
#'   distribution over K classes, use a 1×K matrix. For multiple independent
#'   distributions, use an N×K matrix where each row is a distribution.
#' @param axis The axis along which to sample. Default is -1 (last axis, typically
#'   the class dimension).
#' @param num_samples Number of samples to draw from each distribution.
#' @return An mlx array of integer indices (0-indexed) sampled from the
#'   categorical distributions.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.categorical}
#' @export
#' @examples
#' # Single distribution over 3 classes
#' logits <- matrix(c(0.5, 0.2, 0.3), 1, 3)
#' samples <- mlx_rand_categorical(logits, num_samples = 10)
#'
#' # Multiple distributions
#' logits <- matrix(c(1, 2, 3,
#'                    3, 2, 1), nrow = 2, byrow = TRUE)
#' samples <- mlx_rand_categorical(logits, num_samples = 5)
mlx_rand_categorical <- function(logits, axis = -1L, num_samples = 1L) {
  logits <- as_mlx(logits)
  axis <- as.integer(axis)
  num_samples <- as.integer(num_samples)
  if (num_samples < 1) {
    stop("num_samples must be at least 1.", call. = FALSE)
  }

  ptr <- cpp_mlx_random_categorical(logits, axis, num_samples)
  output_shape <- cpp_mlx_shape(ptr)
  new_mlx(ptr, output_shape, "int32", logits$device)
}

#' Sample random integers on mlx arrays
#'
#' Generates random integers uniformly distributed over the interval [low, high).
#'
#' @inheritParams mlx_creation_params
#' @param low Lower bound (inclusive).
#' @param high Upper bound (exclusive).
#' @param dtype Desired integer dtype ("int32", "int64", "uint32", "uint64").
#' @return An mlx array of random integers.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.randint}
#' @export
#' @examples
#' # Random integers from 0 to 9
#' samples <- mlx_rand_randint(c(3, 3), low = 0, high = 10)
#'
#' # Random integers from -5 to 4
#' samples <- mlx_rand_randint(c(2, 5), low = -5, high = 5)
mlx_rand_randint <- function(dim, low, high,
                             dtype = c("int32", "int64", "uint32", "uint64"),
                             device = mlx_default_device()) {
  dim <- .validate_shape(dim)
  if (!is.numeric(low) || length(low) != 1) {
    stop("low must be a single numeric value.", call. = FALSE)
  }
  if (!is.numeric(high) || length(high) != 1) {
    stop("high must be a single numeric value.", call. = FALSE)
  }
  if (low >= high) {
    stop("low must be less than high.", call. = FALSE)
  }
  low <- as.integer(low)
  high <- as.integer(high)
  dtype <- match.arg(dtype)
  device <- match.arg(device, c("gpu", "cpu"))

  ptr <- cpp_mlx_random_randint(dim, low, high, dtype, device)
  new_mlx(ptr, dim, dtype, device)
}

#' Generate random permutations on mlx arrays
#'
#' Generate a random permutation of integers or permute the entries of an array
#' along a specified axis.
#'
#' @param x Either an integer n (to generate a permutation of 0:(n-1)), or an
#'   mlx array or matrix to permute.
#' @param axis The axis along which to permute when x is an array. Default is 0
#'   (permute rows).
#' @param device Target device ("gpu" or "cpu"). Only used when x is an integer.
#' @return An mlx array containing the random permutation.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.permutation}
#' @export
#' @examples
#' # Generate a random permutation of 0:9
#' perm <- mlx_rand_permutation(10)
#'
#' # Permute the rows of a matrix
#' mat <- matrix(1:12, 4, 3)
#' perm_mat <- mlx_rand_permutation(mat)
#'
#' # Permute columns instead
#' perm_cols <- mlx_rand_permutation(mat, axis = 1)
mlx_rand_permutation <- function(x, axis = 0L, device = mlx_default_device()) {
  axis <- as.integer(axis)

  if (is.numeric(x) && length(x) == 1) {
    # Generate permutation of 0:(x-1)
    n <- as.integer(x)
    if (n < 1) {
      stop("n must be at least 1.", call. = FALSE)
    }
    device <- match.arg(device, c("gpu", "cpu"))
    ptr <- cpp_mlx_random_permutation_n(n, device)
    new_mlx(ptr, n, "int32", device)
  } else {
    # Permute array along axis
    x <- as_mlx(x)
    ptr <- cpp_mlx_random_permutation(x, axis)
    output_shape <- cpp_mlx_shape(ptr)
    new_mlx(ptr, output_shape, x$dtype, x$device)
  }
}

#' Construct MLX random number generator keys
#'
#' `mlx_key()` provides access to MLX's stateless PRNG. Given a 64-bit seed it
#' returns a key that can be passed to other random helpers. Use
#' `mlx_key_split()` to derive multiple independent keys from an existing key.
#'
#' @param seed Integer or numeric seed (converted to unsigned 64-bit).
#' @return An `mlx` array holding the PRNG key.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.key}
#' @export
#' @examples
#' k <- mlx_key(42)
#' subkeys <- mlx_key_split(k, num = 2)
mlx_key <- function(seed) {
  if (length(seed) != 1L || !is.numeric(seed)) {
    stop("`seed` must be a single numeric value.", call. = FALSE)
  }
  ptr <- cpp_mlx_random_key(as.numeric(seed))
  .mlx_wrap_result(ptr, "cpu")
}

#' @rdname mlx_key
#' @param key An `mlx` key array returned by [mlx_key()].
#' @param num Number of subkeys to produce (default 2L).
#' @return A list of `num` `mlx` key arrays.
#' @export
mlx_key_split <- function(key, num = 2L) {
  if (!is.mlx(key)) {
    stop("`key` must be an mlx array produced by mlx_key().", call. = FALSE)
  }
  num <- as.integer(num)
  if (length(num) != 1L || is.na(num) || num <= 0L) {
    stop("`num` must be a positive integer.", call. = FALSE)
  }
  raw <- cpp_mlx_random_split(key$ptr, num)
  lapply(raw, function(ptr) .mlx_wrap_result(ptr, key$device))
}

#' Generate raw random bits on MLX arrays
#'
#' @inheritParams mlx_creation_params
#' @param width Number of bytes per element (default 4 = 32 bits). Must be
#'   positive.
#' @param key Optional `mlx` key array. If omitted, MLX's default generator is
#'   used.
#' @return An `mlx` array of unsigned integers filled with random bits.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.bits}
#' @export
#' @examples
#' k <- mlx_key(12)
#' raw_bits <- mlx_key_bits(c(4, 4), key = k)
mlx_key_bits <- function(dim, width = 4L, key = NULL, device = mlx_default_device()) {
  dim <- .validate_shape(dim)
  width <- as.integer(width)
  if (length(width) != 1L || is.na(width) || width <= 0L) {
    stop("`width` must be a positive integer.", call. = FALSE)
  }
  device <- match.arg(device, c("gpu", "cpu"))

  key_ptr <- if (is.null(key)) {
    NULL
  } else {
    if (!is.mlx(key)) {
      stop("`key` must be an mlx array produced by mlx_key().", call. = FALSE)
    }
    key$ptr
  }

  ptr <- cpp_mlx_random_bits(dim, width, key_ptr, device)
  .mlx_wrap_result(ptr, device)
}
