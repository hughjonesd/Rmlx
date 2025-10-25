#' Create a learnable linear transformation
#'
#' @param in_features Number of input features.
#' @param out_features Number of output features.
#' @param bias Should a bias term be included?
#' @param device Device for the parameters.
#' @return An object of class `mlx_module`.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Linear}
#' @importFrom stats rnorm
#' @export
#' @examples
#' set.seed(1)
#' layer <- mlx_linear(3, 2)
#' x <- as_mlx(matrix(1:6, 2, 3))
#' mlx_forward(layer, x)
mlx_linear <- function(in_features,
                       out_features,
                       bias = TRUE,
                       device = mlx_default_device()) {
  stopifnot(in_features > 0, out_features > 0)

  env <- new.env(parent = emptyenv())
  w_init <- matrix(
    rnorm(in_features * out_features, sd = sqrt(2 / in_features)),
    nrow = in_features,
    ncol = out_features
  )
  env$weight <- as_mlx(w_init, device = device)
  env$bias <- if (bias) {
    as_mlx(matrix(0, nrow = 1, ncol = out_features), device = device)
  } else {
    NULL
  }

  forward <- function(x) {
    y <- x %*% env$weight
    if (!is.null(env$bias)) {
      y <- y + env$bias
    }
    y
  }

  parameters <- function() {
    params <- list(mlx_param(env, "weight"))
    if (!is.null(env$bias)) {
      params <- c(params, list(mlx_param(env, "bias")))
    }
    params
  }

  structure(
    list(
      forward = forward,
      parameters = parameters,
      .env = env
    ),
    class = c("mlx_linear", "mlx_module")
  )
}

#' Rectified linear activation module
#'
#' @return An `mlx_module` applying ReLU.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.ReLU}
#' @export
#' @examples
#' act <- mlx_relu()
#' x <- as_mlx(matrix(c(-1, 0, 2), 3, 1))
#' mlx_forward(act, x)
mlx_relu <- function() {
  forward <- function(x) {
    mask <- x > 0
    mask * x
  }

  structure(
    list(
      forward = forward,
      parameters = function() list()
    ),
    class = c("mlx_relu", "mlx_module")
  )
}

#' Compose modules sequentially
#'
#' @param ... Modules to compose.
#' @return An `mlx_module`.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Sequential}
#' @export
#' @examples
#' set.seed(1)
#' net <- mlx_sequential(mlx_linear(2, 3), mlx_relu(), mlx_linear(3, 1))
#' input <- as_mlx(matrix(c(1, 2), 1, 2))
#' mlx_forward(net, input)
mlx_sequential <- function(...) {
  layers <- list(...)
  if (length(layers) == 0) {
    stop("mlx_sequential() requires at least one module.", call. = FALSE)
  }
  if (!all(vapply(layers, inherits, logical(1), "mlx_module"))) {
    stop("All layers must inherit from `mlx_module`.", call. = FALSE)
  }

  forward <- function(x) {
    for (layer in layers) {
      x <- layer$forward(x)
    }
    x
  }

  parameters <- function() {
    params <- lapply(layers, mlx_parameters)
    unlist(params, recursive = FALSE)
  }

  structure(
    list(
      forward = forward,
      parameters = parameters,
      layers = layers
    ),
    class = c("mlx_sequential", "mlx_module")
  )
}

#' Forward pass utility
#'
#' @param module An `mlx_module`.
#' @param x Input tensor.
#' @return Output tensor.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Module}
#' @export
#' @examples
#' set.seed(1)
#' layer <- mlx_linear(2, 1)
#' input <- as_mlx(matrix(c(1, 2), 1, 2))
#' mlx_forward(layer, input)
mlx_forward <- function(module, x) {
  if (!inherits(module, "mlx_module")) {
    stop("Expected an `mlx_module`.", call. = FALSE)
  }
  module$forward(x)
}

#' Collect parameters from modules
#'
#' @param module An `mlx_module` or list of modules.
#' @return A list of `mlx_param` objects.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Module.parameters}
#' @export
#' @examples
#' set.seed(1)
#' layer <- mlx_linear(2, 1)
#' mlx_parameters(layer)
mlx_parameters <- function(module) {
  if (inherits(module, "mlx_module")) {
    return(module$parameters())
  }
  if (is.list(module)) {
    params <- lapply(module, mlx_parameters)
    return(unlist(params, recursive = FALSE))
  }
  stop("Unsupported type for mlx_parameters().", call. = FALSE)
}

# Internal helper for parameters ----------------------------------------

#' Wrap an environment entry as a learnable parameter
#'
#' @param env Environment storing tensors.
#' @param name Field name within the environment.
#' @return An object of class `mlx_param`.
#' @noRd
mlx_param <- function(env, name) {
  structure(
    list(env = env, name = name),
    class = "mlx_param"
  )
}

#' Retrieve parameter tensors
#'
#' @param params A list of `mlx_param`.
#' @return List of `mlx` tensors.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Module.parameters}
#' @export
#' @examples
#' set.seed(1)
#' layer <- mlx_linear(2, 1)
#' vals <- mlx_param_values(mlx_parameters(layer))
mlx_param_values <- function(params) {
  lapply(params, function(param) {
    stopifnot(inherits(param, "mlx_param"))
    param$env[[param$name]]
  })
}

#' Assign tensors back to parameters
#'
#' @param params A list of `mlx_param`.
#' @param values A list of tensors.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Module.update}
#' @export
#' @examples
#' set.seed(1)
#' layer <- mlx_linear(2, 1)
#' params <- mlx_parameters(layer)
#' current <- mlx_param_values(params)
#' mlx_param_set_values(params, current)
mlx_param_set_values <- function(params, values) {
  stopifnot(length(params) == length(values))
  invisible(Map(function(param, value) {
    stopifnot(inherits(param, "mlx_param"), is.mlx(value))
    param$env[[param$name]] <- value
    NULL
  }, params, values))
}

# Activation functions -------------------------------------------------------

#' GELU activation
#'
#' Gaussian Error Linear Unit activation function.
#'
#' @return An `mlx_module` applying GELU activation.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.GELU}
#' @export
#' @examples
#' act <- mlx_gelu()
#' x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))
#' mlx_forward(act, x)
mlx_gelu <- function() {
  forward <- function(x) {
    # GELU(x) = x * Phi(x) where Phi is the cumulative distribution function of standard normal
    # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x * 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
  }

  structure(
    list(forward = forward, parameters = function() list()),
    class = c("mlx_gelu", "mlx_module")
  )
}

#' Sigmoid activation
#'
#' @return An `mlx_module` applying sigmoid activation.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Sigmoid}
#' @export
#' @examples
#' act <- mlx_sigmoid()
#' x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))
#' mlx_forward(act, x)
mlx_sigmoid <- function() {
  forward <- function(x) {
    1 / (1 + exp(-x))
  }

  structure(
    list(forward = forward, parameters = function() list()),
    class = c("mlx_sigmoid", "mlx_module")
  )
}

#' Tanh activation
#'
#' @return An `mlx_module` applying hyperbolic tangent activation.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Tanh}
#' @export
#' @examples
#' act <- mlx_tanh()
#' x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))
#' mlx_forward(act, x)
mlx_tanh <- function() {
  forward <- function(x) {
    tanh(x)
  }

  structure(
    list(forward = forward, parameters = function() list()),
    class = c("mlx_tanh", "mlx_module")
  )
}

#' Leaky ReLU activation
#'
#' @param negative_slope Slope for negative values (default: 0.01).
#' @return An `mlx_module` applying Leaky ReLU activation.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.LeakyReLU}
#' @export
#' @examples
#' act <- mlx_leaky_relu(negative_slope = 0.1)
#' x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))
#' mlx_forward(act, x)
mlx_leaky_relu <- function(negative_slope = 0.01) {
  stopifnot(negative_slope >= 0)

  forward <- function(x) {
    mlx_maximum(x, negative_slope * x)
  }

  structure(
    list(forward = forward, parameters = function() list()),
    class = c("mlx_leaky_relu", "mlx_module")
  )
}

#' SiLU (Swish) activation
#'
#' Sigmoid Linear Unit, also known as Swish activation.
#'
#' @return An `mlx_module` applying SiLU activation.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.SiLU}
#' @export
#' @examples
#' act <- mlx_silu()
#' x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))
#' mlx_forward(act, x)
mlx_silu <- function() {
  forward <- function(x) {
    x / (1 + exp(-x))
  }

  structure(
    list(forward = forward, parameters = function() list()),
    class = c("mlx_silu", "mlx_module")
  )
}

#' Softmax activation
#'
#' @param axis Axis along which to apply softmax (default: -1, last axis).
#' @return An `mlx_module` applying softmax activation.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Softmax}
#' @export
#' @examples
#' act <- mlx_softmax_layer()
#' x <- as_mlx(matrix(1:6, 2, 3))
#' mlx_forward(act, x)
mlx_softmax_layer <- function(axis = -1L) {
  forward <- function(x) {
    # Convert negative axis to positive (Python convention: -1 = last axis)
    ax <- if (axis < 0) length(x$dim) + axis + 1L else axis
    mlx_softmax(x, axis = ax)
  }

  structure(
    list(forward = forward, parameters = function() list()),
    class = c("mlx_softmax_layer", "mlx_module")
  )
}

# Regularization layers ------------------------------------------------------

#' Dropout layer
#'
#' @param p Probability of dropping an element (default: 0.5).
#' @return An `mlx_module` applying dropout during training.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Dropout}
#' @export
#' @examples
#' set.seed(1)
#' dropout <- mlx_dropout(p = 0.3)
#' x <- as_mlx(matrix(1:12, 3, 4))
#' mlx_forward(dropout, x)
mlx_dropout <- function(p = 0.5) {
  stopifnot(p >= 0 && p <= 1)

  env <- new.env(parent = emptyenv())
  env$training <- TRUE
  env$p <- p

  forward <- function(x) {
    if (!env$training || env$p == 0) {
      return(x)
    }
    if (env$p == 1) {
      return(x * 0)
    }
    # Generate dropout mask
    mask <- mlx_rand_bernoulli(x$dim, prob = 1 - env$p, device = x$device)
    # Scale by 1/(1-p) to maintain expected value
    x * mask / (1 - env$p)
  }

  set_training <- function(mode = TRUE) {
    env$training <- mode
  }

  structure(
    list(
      forward = forward,
      parameters = function() list(),
      set_training = set_training,
      .env = env
    ),
    class = c("mlx_dropout", "mlx_module")
  )
}

# Normalization layers -------------------------------------------------------

#' Layer normalization
#'
#' Normalizes inputs across the feature dimension.
#'
#' @param normalized_shape Size of the feature dimension to normalize.
#' @param eps Small constant for numerical stability (default: 1e-5).
#' @param device Device for parameters.
#' @return An `mlx_module` applying layer normalization.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.LayerNorm}
#' @export
#' @examples
#' set.seed(1)
#' ln <- mlx_layer_norm(4)
#' x <- as_mlx(matrix(rnorm(12), 3, 4))
#' mlx_forward(ln, x)
mlx_layer_norm <- function(normalized_shape, eps = 1e-5, device = mlx_default_device()) {
  stopifnot(normalized_shape > 0)
  stopifnot(eps > 0)

  env <- new.env(parent = emptyenv())
  env$gamma <- as_mlx(rep(1, normalized_shape), device = device)
  env$beta <- as_mlx(rep(0, normalized_shape), device = device)
  env$eps <- eps

  forward <- function(x) {
    # Normalize across last dimension
    ndim <- length(x$dim)
    last_axis <- ndim
    mean_x <- mlx_mean(x, axis = last_axis, drop = FALSE)
    var_x <- mlx_var(x, axis = last_axis, drop = FALSE, ddof = 0L)

    # Normalize
    x_norm <- (x - mean_x) / sqrt(var_x + env$eps)

    # Scale and shift
    x_norm * env$gamma + env$beta
  }

  parameters <- function() {
    list(
      mlx_param(env, "gamma"),
      mlx_param(env, "beta")
    )
  }

  structure(
    list(
      forward = forward,
      parameters = parameters,
      .env = env
    ),
    class = c("mlx_layer_norm", "mlx_module")
  )
}

#' Batch normalization
#'
#' Normalizes inputs across the batch dimension.
#'
#' @param num_features Number of feature channels.
#' @param eps Small constant for numerical stability (default: 1e-5).
#' @param momentum Momentum for running statistics (default: 0.1).
#' @param device Device for parameters.
#' @return An `mlx_module` applying batch normalization.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.BatchNorm}
#' @export
#' @examples
#' set.seed(1)
#' bn <- mlx_batch_norm(4)
#' x <- as_mlx(matrix(rnorm(12), 3, 4))
#' mlx_forward(bn, x)
mlx_batch_norm <- function(num_features, eps = 1e-5, momentum = 0.1, device = mlx_default_device()) {
  stopifnot(num_features > 0)
  stopifnot(eps > 0)
  stopifnot(momentum >= 0 && momentum <= 1)

  env <- new.env(parent = emptyenv())
  env$gamma <- as_mlx(rep(1, num_features), device = device)
  env$beta <- as_mlx(rep(0, num_features), device = device)
  env$running_mean <- as_mlx(rep(0, num_features), device = device)
  env$running_var <- as_mlx(rep(1, num_features), device = device)
  env$eps <- eps
  env$momentum <- momentum
  env$training <- TRUE

  forward <- function(x) {
    if (env$training) {
      # Compute batch statistics
      batch_mean <- mlx_mean(x, axis = 1L, drop = FALSE)
      batch_var <- mlx_var(x, axis = 1L, drop = FALSE, ddof = 0L)

      # Update running statistics
      env$running_mean <- (1 - env$momentum) * env$running_mean + env$momentum * batch_mean
      env$running_var <- (1 - env$momentum) * env$running_var + env$momentum * batch_var

      mean_to_use <- batch_mean
      var_to_use <- batch_var
    } else {
      mean_to_use <- env$running_mean
      var_to_use <- env$running_var
    }

    # Normalize
    x_norm <- (x - mean_to_use) / sqrt(var_to_use + env$eps)

    # Scale and shift
    x_norm * env$gamma + env$beta
  }

  parameters <- function() {
    list(
      mlx_param(env, "gamma"),
      mlx_param(env, "beta")
    )
  }

  set_training <- function(mode = TRUE) {
    env$training <- mode
  }

  structure(
    list(
      forward = forward,
      parameters = parameters,
      set_training = set_training,
      .env = env
    ),
    class = c("mlx_batch_norm", "mlx_module")
  )
}

# Embedding layer ------------------------------------------------------------

#' Embedding layer
#'
#' Maps discrete tokens to continuous vectors.
#'
#' @param num_embeddings Size of vocabulary.
#' @param embedding_dim Dimension of embedding vectors.
#' @param device Device for parameters.
#' @return An `mlx_module` for token embeddings.
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Embedding}
#' @export
#' @examples
#' set.seed(1)
#' emb <- mlx_embedding(num_embeddings = 100, embedding_dim = 16)
#' # Token indices (0-indexed)
#' tokens <- as_mlx(matrix(c(5, 10, 3, 7), 2, 2))
#' mlx_forward(emb, tokens)
mlx_embedding <- function(num_embeddings, embedding_dim, device = mlx_default_device()) {
  stopifnot(num_embeddings > 0, embedding_dim > 0)

  env <- new.env(parent = emptyenv())
  # Initialize with small random values
  weight_init <- matrix(
    rnorm(num_embeddings * embedding_dim, sd = 0.01),
    num_embeddings,
    embedding_dim
  )
  env$weight <- as_mlx(weight_init, device = device)
  env$num_embeddings <- num_embeddings
  env$embedding_dim <- embedding_dim

  forward <- function(indices) {
    if (!is.mlx(indices)) indices <- as_mlx(indices)

    # indices are 0-based token IDs
    orig_shape <- indices$dim
    indices_r <- as.integer(as.matrix(indices))

    # Take embeddings
    result_list <- lapply(indices_r, function(idx) {
      if (idx < 0 || idx >= env$num_embeddings) {
        stop("Index out of range: ", idx, call. = FALSE)
      }
      as.numeric(as.matrix(env$weight[idx + 1, ]))
    })

    # Stack results and reshape
    if (length(orig_shape) == 0) {
      # Scalar index - return (1, embedding_dim)
      as_mlx(matrix(result_list[[1]], 1, env$embedding_dim), device = indices$device)
    } else {
      # Stack into matrix then reshape to match input shape + embedding dimension
      result_mat <- do.call(rbind, lapply(result_list, function(x) matrix(x, 1, env$embedding_dim)))
      new_shape <- c(orig_shape, env$embedding_dim)
      result_array <- array(as.numeric(result_mat), dim = new_shape)
      as_mlx(result_array, device = indices$device)
    }
  }

  parameters <- function() {
    list(mlx_param(env, "weight"))
  }

  structure(
    list(
      forward = forward,
      parameters = parameters,
      .env = env
    ),
    class = c("mlx_embedding", "mlx_module")
  )
}

#' 1D Convolution
#'
#' Applies a 1D convolution over the input signal.
#'
#' @param input Input array of shape `(N, L, C_in)` where N is batch size,
#'   L is sequence length, and C_in is number of input channels
#' @param weight Weight tensor of shape `(C_out, kernel_size, C_in)`
#' @param stride Stride of the convolution (default: 1)
#' @param padding Amount of zero padding (default: 0)
#' @param dilation Spacing between kernel elements (default: 1)
#' @param groups Number of blocked connections (default: 1)
#' @param device Device to use for computation
#' @return Convolved output array
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv1d.html}
#' @export
mlx_conv1d <- function(input, weight, stride = 1L, padding = 0L, dilation = 1L,
                       groups = 1L, device = mlx_default_device()) {
  if (!is.mlx(input)) input <- as_mlx(input)
  if (!is.mlx(weight)) weight <- as_mlx(weight)

  ptr <- cpp_mlx_conv1d(input$ptr, weight$ptr, as.integer(stride),
                       as.integer(padding), as.integer(dilation),
                       as.integer(groups), device)
  .mlx_wrap_result(ptr, device)
}

#' 2D Convolution
#'
#' Applies a 2D convolution over the input image.
#'
#' @param input Input array of shape `(N, H, W, C_in)` where N is batch size,
#'   H and W are height and width, and C_in is number of input channels
#' @param weight Weight tensor of shape `(C_out, kernel_h, kernel_w, C_in)`
#' @param stride Stride of the convolution. Can be a single integer or a vector of
#'   length 2 for (stride_h, stride_w). Default: c(1, 1)
#' @param padding Amount of zero padding. Can be a single integer or a vector of
#'   length 2 for (padding_h, padding_w). Default: c(0, 0)
#' @param dilation Spacing between kernel elements. Can be a single integer or a vector of
#'   length 2 for (dilation_h, dilation_w). Default: c(1, 1)
#' @param groups Number of blocked connections from input to output channels (default: 1)
#' @param device Device to use for computation
#' @return Convolved output array
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv2d.html}
#' @export
#' @examples
#' # Create a simple 2D convolution
#' input <- as_mlx(array(rnorm(1*28*28*3), dim = c(1, 28, 28, 3)))  # Batch of 1 RGB image
#' weight <- as_mlx(array(rnorm(16*3*3*3), dim = c(16, 3, 3, 3)))  # 16 filters, 3x3 kernel
#' output <- mlx_conv2d(input, weight, stride = c(1, 1), padding = c(1, 1))
mlx_conv2d <- function(input, weight, stride = c(1L, 1L), padding = c(0L, 0L),
                       dilation = c(1L, 1L), groups = 1L,
                       device = mlx_default_device()) {
  if (!is.mlx(input)) input <- as_mlx(input)
  if (!is.mlx(weight)) weight <- as_mlx(weight)

  # Handle scalar inputs
  if (length(stride) == 1) stride <- rep(stride, 2)
  if (length(padding) == 1) padding <- rep(padding, 2)
  if (length(dilation) == 1) dilation <- rep(dilation, 2)

  ptr <- cpp_mlx_conv2d(input$ptr, weight$ptr, as.integer(stride),
                       as.integer(padding), as.integer(dilation),
                       as.integer(groups), device)
  .mlx_wrap_result(ptr, device)
}

#' 3D Convolution
#'
#' Applies a 3D convolution over the input volume.
#'
#' @param input Input array of shape `(N, D, H, W, C_in)` where N is batch size,
#'   D, H, W are depth, height and width, and C_in is number of input channels
#' @param weight Weight tensor of shape `(C_out, kernel_d, kernel_h, kernel_w, C_in)`
#' @param stride Stride of the convolution. Can be a single integer or a vector of
#'   length 3 for (stride_d, stride_h, stride_w). Default: c(1, 1, 1)
#' @param padding Amount of zero padding. Can be a single integer or a vector of
#'   length 3 for (padding_d, padding_h, padding_w). Default: c(0, 0, 0)
#' @param dilation Spacing between kernel elements. Can be a single integer or a vector of
#'   length 3 for (dilation_d, dilation_h, dilation_w). Default: c(1, 1, 1)
#' @param groups Number of blocked connections from input to output channels (default: 1)
#' @param device Device to use for computation
#' @return Convolved output array
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv3d.html}
#' @export
mlx_conv3d <- function(input, weight, stride = c(1L, 1L, 1L), padding = c(0L, 0L, 0L),
                       dilation = c(1L, 1L, 1L), groups = 1L,
                       device = mlx_default_device()) {
  if (!is.mlx(input)) input <- as_mlx(input)
  if (!is.mlx(weight)) weight <- as_mlx(weight)

  # Handle scalar inputs
  if (length(stride) == 1) stride <- rep(stride, 3)
  if (length(padding) == 1) padding <- rep(padding, 3)
  if (length(dilation) == 1) dilation <- rep(dilation, 3)

  ptr <- cpp_mlx_conv3d(input$ptr, weight$ptr, as.integer(stride),
                       as.integer(padding), as.integer(dilation),
                       as.integer(groups), device)
  .mlx_wrap_result(ptr, device)
}

#' 1D Transposed Convolution
#'
#' Applies a 1D transposed convolution (also called deconvolution) over an input signal.
#' Transposed convolutions are used to upsample the spatial dimensions of the input.
#'
#' @param input An mlx array with shape `(batch, length, in_channels)` for 'NWC' layout
#' @param weight An mlx array with shape `(out_channels, kernel_size, in_channels)`
#' @param stride Stride of the convolution. Default: 1
#' @param padding Amount of zero padding. Default: 0
#' @param dilation Dilation factor for the kernel. Default: 1
#' @param output_padding Additional size added to output shape. Default: 0
#' @param groups Number of blocked connections from input to output channels. Default: 1
#' @param device Device to perform computation on. Default: `mlx_default_device()`
#'
#' @return An mlx array with the transposed convolution result
#' @seealso [mlx_conv1d()], [mlx_conv_transpose2d()], [mlx_conv_transpose3d()]
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html}
#' @export
mlx_conv_transpose1d <- function(input, weight, stride = 1L, padding = 0L,
                                  dilation = 1L, output_padding = 0L, groups = 1L,
                                  device = mlx_default_device()) {
  if (!is.mlx(input)) input <- as_mlx(input)
  if (!is.mlx(weight)) weight <- as_mlx(weight)

  ptr <- cpp_mlx_conv_transpose1d(input$ptr, weight$ptr, as.integer(stride),
                                   as.integer(padding), as.integer(dilation),
                                   as.integer(output_padding), as.integer(groups),
                                   device)
  .mlx_wrap_result(ptr, device)
}

#' 2D Transposed Convolution
#'
#' Applies a 2D transposed convolution (also called deconvolution) over an input signal.
#' Transposed convolutions are commonly used in image generation and upsampling tasks.
#'
#' @param input An mlx array with shape `(batch, height, width, in_channels)` for 'NHWC' layout
#' @param weight An mlx array with shape `(out_channels, kernel_h, kernel_w, in_channels)`
#' @param stride Stride of the convolution. Can be a scalar or length-2 vector. Default: c(1, 1)
#' @param padding Amount of zero padding. Can be a scalar or length-2 vector. Default: c(0, 0)
#' @param dilation Dilation factor for the kernel. Can be a scalar or length-2 vector. Default: c(1, 1)
#' @param output_padding Additional size added to output shape. Can be a scalar or length-2 vector. Default: c(0, 0)
#' @param groups Number of blocked connections from input to output channels. Default: 1
#' @param device Device to perform computation on. Default: `mlx_default_device()`
#'
#' @return An mlx array with the transposed convolution result
#' @seealso [mlx_conv2d()], [mlx_conv_transpose1d()], [mlx_conv_transpose3d()]
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html}
#' @export
mlx_conv_transpose2d <- function(input, weight, stride = c(1L, 1L),
                                  padding = c(0L, 0L), dilation = c(1L, 1L),
                                  output_padding = c(0L, 0L), groups = 1L,
                                  device = mlx_default_device()) {
  if (!is.mlx(input)) input <- as_mlx(input)
  if (!is.mlx(weight)) weight <- as_mlx(weight)

  # Handle scalar inputs
  if (length(stride) == 1) stride <- rep(stride, 2)
  if (length(padding) == 1) padding <- rep(padding, 2)
  if (length(dilation) == 1) dilation <- rep(dilation, 2)
  if (length(output_padding) == 1) output_padding <- rep(output_padding, 2)

  ptr <- cpp_mlx_conv_transpose2d(input$ptr, weight$ptr, as.integer(stride),
                                   as.integer(padding), as.integer(dilation),
                                   as.integer(output_padding), as.integer(groups),
                                   device)
  .mlx_wrap_result(ptr, device)
}

#' 3D Transposed Convolution
#'
#' Applies a 3D transposed convolution (also called deconvolution) over an input signal.
#' Useful for 3D volumetric data upsampling, such as in medical imaging or video generation.
#'
#' @param input An mlx array with shape `(batch, depth, height, width, in_channels)` for 'NDHWC' layout
#' @param weight An mlx array with shape `(out_channels, kernel_d, kernel_h, kernel_w, in_channels)`
#' @param stride Stride of the convolution. Can be a scalar or length-3 vector. Default: c(1, 1, 1)
#' @param padding Amount of zero padding. Can be a scalar or length-3 vector. Default: c(0, 0, 0)
#' @param dilation Dilation factor for the kernel. Can be a scalar or length-3 vector. Default: c(1, 1, 1)
#' @param output_padding Additional size added to output shape. Can be a scalar or length-3 vector. Default: c(0, 0, 0)
#' @param groups Number of blocked connections from input to output channels. Default: 1
#' @param device Device to perform computation on. Default: `mlx_default_device()`
#'
#' @return An mlx array with the transposed convolution result
#' @seealso [mlx_conv3d()], [mlx_conv_transpose1d()], [mlx_conv_transpose2d()]
#' @seealso \url{https://ml-explore.github.io/mlx/build/html/python/nn.html}
#' @export
mlx_conv_transpose3d <- function(input, weight, stride = c(1L, 1L, 1L),
                                  padding = c(0L, 0L, 0L), dilation = c(1L, 1L, 1L),
                                  output_padding = c(0L, 0L, 0L), groups = 1L,
                                  device = mlx_default_device()) {
  if (!is.mlx(input)) input <- as_mlx(input)
  if (!is.mlx(weight)) weight <- as_mlx(weight)

  # Handle scalar inputs
  if (length(stride) == 1) stride <- rep(stride, 3)
  if (length(padding) == 1) padding <- rep(padding, 3)
  if (length(dilation) == 1) dilation <- rep(dilation, 3)
  if (length(output_padding) == 1) output_padding <- rep(output_padding, 3)

  ptr <- cpp_mlx_conv_transpose3d(input$ptr, weight$ptr, as.integer(stride),
                                   as.integer(padding), as.integer(dilation),
                                   as.integer(output_padding), as.integer(groups),
                                   device)
  .mlx_wrap_result(ptr, device)
}
