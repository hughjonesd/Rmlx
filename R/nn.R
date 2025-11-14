# Internal utilities ---------------------------------------------------------

#' @title Internal helpers for mlx modules
#' @description Constructors and utilities shared across module definitions.
#' @noRd
new_mlx_module <- function(forward,
                           parameters = function() list(),
                           set_training = NULL,
                           fields = list(),
                           classes = character()) {
  stopifnot(is.function(forward), is.function(parameters))
  if (!is.null(set_training)) {
    stopifnot(is.function(set_training))
  }

  module <- c(list(forward = forward, parameters = parameters), fields)
  if (!is.null(set_training)) {
    module$set_training <- set_training
  }

  class(module) <- unique(c(classes, "mlx_module"))
  module
}

mlx_module_forward <- function(module, x) {
  if (!inherits(module, "mlx_module")) {
    stop("Expected an `mlx_module`.", call. = FALSE)
  }
  module$forward(x)
}

mlx_module_parameters <- function(module) {
  if (!inherits(module, "mlx_module")) {
    stop("Expected an `mlx_module`.", call. = FALSE)
  }
  module$parameters()
}

mlx_module_set_training <- function(module, mode = TRUE) {
  if (!inherits(module, "mlx_module")) {
    stop("Expected an `mlx_module`.", call. = FALSE)
  }
  setter <- module$set_training
  if (is.null(setter)) {
    return(invisible(module))
  }
  setter(mode)
  invisible(module)
}

#' Create a learnable linear transformation
#'
#' @param in_features Number of input features.
#' @param out_features Number of output features.
#' @param bias Should a bias term be included?
#' @inheritParams common_params
#' @return An object of class `mlx_module`.
#' @seealso [mlx.nn.Linear](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Linear)
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

  new_mlx_module(
    forward = forward,
    parameters = parameters,
    fields = list(.env = env),
    classes = "mlx_linear"
  )
}

#' Rectified linear activation module
#'
#' @return An `mlx_module` applying ReLU.
#' @seealso [mlx.nn.ReLU](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.ReLU)
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

  new_mlx_module(
    forward = forward,
    parameters = function() list(),
    classes = "mlx_relu"
  )
}

#' Compose modules sequentially
#'
#' @param ... Modules to compose.
#' @return An `mlx_module`.
#' @seealso [mlx.nn.Sequential](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Sequential)
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
      x <- mlx_module_forward(layer, x)
    }
    x
  }

  parameters <- function() {
    params <- lapply(layers, mlx_module_parameters)
    unlist(params, recursive = FALSE)
  }

  set_training <- function(mode = TRUE) {
    lapply(layers, mlx_module_set_training, mode = mode)
    invisible(NULL)
  }

  new_mlx_module(
    forward = forward,
    parameters = parameters,
    set_training = set_training,
    fields = list(layers = layers),
    classes = "mlx_sequential"
  )
}

#' Forward pass utility
#'
#' @param module An `mlx_module`.
#' @inheritParams mlx_array_required
#' @return Output array.
#' @seealso [mlx.nn.Module](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Module)
#' @export
#' @examples
#' set.seed(1)
#' layer <- mlx_linear(2, 1)
#' input <- as_mlx(matrix(c(1, 2), 1, 2))
#' mlx_forward(layer, input)
mlx_forward <- function(module, x) {
  mlx_module_forward(module, x)
}

#' Collect parameters from modules
#'
#' @param module An `mlx_module` or list of modules.
#' @return A list of `mlx_param` objects.
#' @seealso [mlx.nn.Module.parameters](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Module.parameters)
#' @export
#' @examples
#' set.seed(1)
#' layer <- mlx_linear(2, 1)
#' mlx_parameters(layer)
mlx_parameters <- function(module) {
  if (inherits(module, "mlx_module")) {
    return(mlx_module_parameters(module))
  }
  if (is.list(module)) {
    params <- lapply(module, mlx_parameters)
    return(unlist(params, recursive = FALSE))
  }
  stop("Unsupported type for mlx_parameters().", call. = FALSE)
}

#' Toggle training mode for MLX modules
#'
#' `mlx_set_training()` switches modules between training and evaluation modes.
#' Layers that do not implement training-specific behaviour ignore the call.
#'
#' @param module An object inheriting from `mlx_module`.
#' @param mode Logical flag; `TRUE` for training mode, `FALSE` for evaluation.
#' @return The input module (invisibly).
#' @seealso <https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Module.train>
#' @export
#' @examples
#' model <- mlx_sequential(mlx_linear(2, 4), mlx_dropout(0.5))
#' mlx_set_training(model, FALSE)
mlx_set_training <- function(module, mode = TRUE) {
  if (!inherits(module, "mlx_module")) {
    stop("Expected an `mlx_module`.", call. = FALSE)
  }
  mlx_module_set_training(module, mode = mode)
  invisible(module)
}

# Internal helper for parameters ----------------------------------------

#' Wrap an environment entry as a learnable parameter
#'
#' @param env Environment storing arrays.
#' @param name Field name within the environment.
#' @return An object of class `mlx_param`.
#' @noRd
mlx_param <- function(env, name) {
  structure(
    list(env = env, name = name),
    class = "mlx_param"
  )
}

#' Retrieve parameter arrays
#'
#' @param params A list of `mlx_param`.
#' @return List of mlx arrays.
#' @seealso [mlx.nn.Module.parameters](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Module.parameters)
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

#' Assign arrays back to parameters
#'
#' @param params A list of `mlx_param`.
#' @param values A list of arrays.
#' @seealso [mlx.nn.Module.update](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Module.update)
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
#' @seealso [mlx.nn.GELU](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.GELU)
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

  new_mlx_module(
    forward = forward,
    parameters = function() list(),
    classes = "mlx_gelu"
  )
}

#' Sigmoid activation
#'
#' @return An `mlx_module` applying sigmoid activation.
#' @seealso [mlx.nn.Sigmoid](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Sigmoid)
#' @export
#' @examples
#' act <- mlx_sigmoid()
#' x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))
#' mlx_forward(act, x)
mlx_sigmoid <- function() {
  forward <- function(x) {
    1 / (1 + exp(-x))
  }

  new_mlx_module(
    forward = forward,
    parameters = function() list(),
    classes = "mlx_sigmoid"
  )
}

#' Tanh activation
#'
#' @return An `mlx_module` applying hyperbolic tangent activation.
#' @seealso [mlx.nn.Tanh](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Tanh)
#' @export
#' @examples
#' act <- mlx_tanh()
#' x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))
#' mlx_forward(act, x)
mlx_tanh <- function() {
  forward <- function(x) {
    tanh(x)
  }

  new_mlx_module(
    forward = forward,
    parameters = function() list(),
    classes = "mlx_tanh"
  )
}

#' Leaky ReLU activation
#'
#' @param negative_slope Slope for negative values (default: 0.01).
#' @return An `mlx_module` applying Leaky ReLU activation.
#' @seealso [mlx.nn.LeakyReLU](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.LeakyReLU)
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

  new_mlx_module(
    forward = forward,
    parameters = function() list(),
    classes = "mlx_leaky_relu"
  )
}

#' SiLU (Swish) activation
#'
#' Sigmoid Linear Unit, also known as Swish activation.
#'
#' @return An `mlx_module` applying SiLU activation.
#' @seealso [mlx.nn.SiLU](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.SiLU)
#' @export
#' @examples
#' act <- mlx_silu()
#' x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))
#' mlx_forward(act, x)
mlx_silu <- function() {
  forward <- function(x) {
    x / (1 + exp(-x))
  }

  new_mlx_module(
    forward = forward,
    parameters = function() list(),
    classes = "mlx_silu"
  )
}

#' Softmax activation
#'
#' @param axis Axis along which to apply softmax (default: -1, last axis).
#' @return An `mlx_module` applying softmax activation.
#' @seealso [mlx.nn.Softmax](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Softmax)
#' @export
#' @examples
#' act <- mlx_softmax_layer()
#' x <- as_mlx(matrix(1:6, 2, 3))
#' mlx_forward(act, x)
mlx_softmax_layer <- function(axis = -1L) {
  forward <- function(x) {
    # Convert negative axis to positive (Python convention: -1 = last axis)
    ax <- if (axis < 0) length(dim(x)) + axis + 1L else axis
    mlx_softmax(x, axis = ax)
  }

  new_mlx_module(
    forward = forward,
    parameters = function() list(),
    classes = "mlx_softmax_layer"
  )
}

# Regularization layers ------------------------------------------------------

#' Dropout layer
#'
#' @param p Probability of dropping an element (default: 0.5).
#' @return An `mlx_module` applying dropout during training.
#' @seealso [mlx.nn.Dropout](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Dropout)
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
    mask <- mlx_rand_bernoulli(dim(x), prob = 1 - env$p, device = x$device)
    # Scale by 1/(1-p) to maintain expected value
    x * mask / (1 - env$p)
  }

  set_training <- function(mode = TRUE) {
    env$training <- mode
  }

  new_mlx_module(
    forward = forward,
    parameters = function() list(),
    set_training = set_training,
    fields = list(.env = env),
    classes = "mlx_dropout"
  )
}

# Normalization layers -------------------------------------------------------

#' Layer normalization
#'
#' Normalizes inputs across the feature dimension.
#'
#' @param normalized_shape Size of the feature dimension to normalize.
#' @param eps Small constant for numerical stability (default: 1e-5).
#' @inheritParams common_params
#' @return An `mlx_module` applying layer normalization.
#' @seealso [mlx.nn.LayerNorm](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.LayerNorm)
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
    ndim <- length(dim(x))
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

  new_mlx_module(
    forward = forward,
    parameters = parameters,
    fields = list(.env = env),
    classes = "mlx_layer_norm"
  )
}

#' Batch normalization
#'
#' Normalizes inputs across the batch dimension.
#'
#' @param num_features Number of feature channels.
#' @param eps Small constant for numerical stability (default: 1e-5).
#' @param momentum Momentum for running statistics (default: 0.1).
#' @inheritParams common_params
#' @return An `mlx_module` applying batch normalization.
#' @seealso [mlx.nn.BatchNorm](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.BatchNorm)
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

  new_mlx_module(
    forward = forward,
    parameters = parameters,
    set_training = set_training,
    fields = list(.env = env),
    classes = "mlx_batch_norm"
  )
}

# Embedding layer ------------------------------------------------------------

#' Embedding layer
#'
#' Maps discrete tokens to continuous vectors.
#'
#' @param num_embeddings Size of vocabulary.
#' @param embedding_dim Dimension of embedding vectors.
#' @inheritParams common_params
#' @return An `mlx_module` for token embeddings.
#' @seealso [mlx.nn.Embedding](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Embedding)
#' @export
#' @examples
#' set.seed(1)
#' emb <- mlx_embedding(num_embeddings = 100, embedding_dim = 16)
#' # Token indices (1-indexed)
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
    indices <- as_mlx(indices)

    # indices are 1-based token IDs
    orig_shape <- dim(indices)
    indices_r <- as.integer(as.matrix(indices))

    # Take embeddings
    result_list <- lapply(indices_r, function(idx) {
      if (idx < 1 || idx > env$num_embeddings) {
        stop("Index out of range: ", idx, call. = FALSE)
      }
      as.numeric(as.matrix(env$weight[idx, ]))
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

  new_mlx_module(
    forward = forward,
    parameters = parameters,
    fields = list(.env = env),
    classes = "mlx_embedding"
  )
}

#' 1D Convolution
#'
#' Applies a 1D convolution over the input signal.
#'
#' Input has shape `(N, L, C_in)` where N is batch size, L is sequence length,
#' and C_in is number of input channels. Weight has shape `(C_out, kernel_size, C_in)`.
#'
#' @inheritParams conv_params
#' @inheritParams common_params
#' @return Convolved output array
#' @seealso [mlx.core.conv1d](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv1d.html)
#' @export
mlx_conv1d <- function(input, weight, stride = 1L, padding = 0L, dilation = 1L,
                       groups = 1L, device = mlx_default_device()) {
  input <- as_mlx(input)
  weight <- as_mlx(weight)

  ptr <- cpp_mlx_conv1d(input$ptr, weight$ptr, as.integer(stride),
                       as.integer(padding), as.integer(dilation),
                       as.integer(groups), device)
  .mlx_wrap_result(ptr, device)
}

#' 2D Convolution
#'
#' Applies a 2D convolution over the input image.
#'
#' Input has shape `(N, H, W, C_in)` where N is batch size, H and W are height
#' and width, and C_in is number of input channels. Weight has shape
#' `(C_out, kernel_h, kernel_w, C_in)`.
#'
#' @inheritParams conv_params
#' @inheritParams common_params
#' @return Convolved output array
#' @seealso [mlx.core.conv2d](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv2d.html)
#' @export
#' @examples
#' # Create a simple 2D convolution
#' input <- as_mlx(array(rnorm(1*28*28*3), dim = c(1, 28, 28, 3)))  # Batch of 1 RGB image
#' weight <- as_mlx(array(rnorm(16*3*3*3), dim = c(16, 3, 3, 3)))  # 16 filters, 3x3 kernel
#' output <- mlx_conv2d(input, weight, stride = c(1, 1), padding = c(1, 1))
mlx_conv2d <- function(input, weight, stride = c(1L, 1L), padding = c(0L, 0L),
                       dilation = c(1L, 1L), groups = 1L,
                       device = mlx_default_device()) {
  input <- as_mlx(input)
  weight <- as_mlx(weight)

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
#' Input has shape `(N, D, H, W, C_in)` where N is batch size, D, H, W are depth,
#' height and width, and C_in is number of input channels. Weight has shape
#' `(C_out, kernel_d, kernel_h, kernel_w, C_in)`.
#'
#' @inheritParams conv_params
#' @inheritParams common_params
#' @return Convolved output array
#' @seealso [mlx.core.conv3d](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv3d.html)
#' @export
mlx_conv3d <- function(input, weight, stride = c(1L, 1L, 1L), padding = c(0L, 0L, 0L),
                       dilation = c(1L, 1L, 1L), groups = 1L,
                       device = mlx_default_device()) {
  input <- as_mlx(input)
  weight <- as_mlx(weight)

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
#' Input has shape `(batch, length, in_channels)` for 'NWC' layout. Weight has shape
#' `(out_channels, kernel_size, in_channels)`.
#'
#' @inheritParams conv_params
#' @param output_padding Additional size added to output shape. Default: 0
#' @inheritParams common_params
#'
#' @return An mlx array with the transposed convolution result
#' @seealso [mlx_conv1d()], [mlx_conv_transpose2d()], [mlx_conv_transpose3d()]
#' @seealso [mlx.nn](https://ml-explore.github.io/mlx/build/html/python/nn.html)
#' @export
mlx_conv_transpose1d <- function(input, weight, stride = 1L, padding = 0L,
                                  dilation = 1L, output_padding = 0L, groups = 1L,
                                  device = mlx_default_device()) {
  input <- as_mlx(input)
  weight <- as_mlx(weight)

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
#' Input has shape `(batch, height, width, in_channels)` for 'NHWC' layout. Weight has
#' shape `(out_channels, kernel_h, kernel_w, in_channels)`.
#'
#' @inheritParams conv_params
#' @param output_padding Additional size added to output shape. Can be a scalar or
#'   length-2 vector. Default: c(0, 0)
#' @inheritParams common_params
#'
#' @return An mlx array with the transposed convolution result
#' @seealso [mlx_conv2d()], [mlx_conv_transpose1d()], [mlx_conv_transpose3d()]
#' @seealso [mlx.nn](https://ml-explore.github.io/mlx/build/html/python/nn.html)
#' @export
mlx_conv_transpose2d <- function(input, weight, stride = c(1L, 1L),
                                  padding = c(0L, 0L), dilation = c(1L, 1L),
                                  output_padding = c(0L, 0L), groups = 1L,
                                  device = mlx_default_device()) {
  input <- as_mlx(input)
  weight <- as_mlx(weight)

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
#' Input has shape `(batch, depth, height, width, in_channels)` for 'NDHWC' layout.
#' Weight has shape `(out_channels, kernel_d, kernel_h, kernel_w, in_channels)`.
#'
#' @inheritParams conv_params
#' @param output_padding Additional size added to output shape. Can be a scalar or
#'   length-3 vector. Default: c(0, 0, 0)
#' @inheritParams common_params
#'
#' @return An mlx array with the transposed convolution result
#' @seealso [mlx_conv3d()], [mlx_conv_transpose1d()], [mlx_conv_transpose2d()]
#' @seealso [mlx.nn](https://ml-explore.github.io/mlx/build/html/python/nn.html)
#' @export
mlx_conv_transpose3d <- function(input, weight, stride = c(1L, 1L, 1L),
                                  padding = c(0L, 0L, 0L), dilation = c(1L, 1L, 1L),
                                  output_padding = c(0L, 0L, 0L), groups = 1L,
                                  device = mlx_default_device()) {
  input <- as_mlx(input)
  weight <- as_mlx(weight)

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

#' Quantize a Matrix
#'
#' Quantizes a weight matrix to low-precision representation (typically 4-bit or 8-bit).
#' This reduces memory usage and enables faster computation during inference.
#'
#' @param w An mlx array (the weight matrix to quantize)
#' @param group_size The group size for quantization. Smaller groups provide better
#'   accuracy but slightly higher memory. Default: 64
#' @param bits The number of bits for quantization (typically 4 or 8). Default: 4
#' @param mode The quantization mode: "affine" (with scales and biases) or "mxfp4"
#'   (4-bit floating point with group_size=32). Default: "affine"
#' @inheritParams common_params
#'
#' @return A list containing:
#'   \item{w_q}{The quantized weight matrix (packed as uint32)}
#'   \item{scales}{The quantization scales for dequantization}
#'   \item{biases}{The quantization biases (NULL for symmetric mode)}
#'
#' @details
#' Quantization converts floating-point weights to low-precision integers, reducing
#' memory by up to 8x for 4-bit quantization. The scales (and optionally biases) are
#' stored to enable approximate reconstruction of the original values.
#'
#' @examples
#' w <- mlx_rand_normal(c(64, 32))
#' quant <- mlx_quantize(w, group_size = 32, bits = 4)
#' # Use quant$w_q, quant$scales, quant$biases with mlx_quantized_matmul()
#'
#' @seealso [mlx_dequantize()], [mlx_quantized_matmul()]
#' @export
mlx_quantize <- function(w, group_size = 64L, bits = 4L, mode = "affine",
                         device = mlx_default_device()) {
  w <- as_mlx(w)

  result <- cpp_mlx_quantize(w$ptr, as.integer(group_size), as.integer(bits),
                              mode, device)

  # Wrap the returned pointers as mlx objects
  out <- list()
  out$w_q <- .mlx_wrap_result(result$w_q, device)
  out$scales <- .mlx_wrap_result(result$scales, device)
  if (!is.null(result$biases)) {
    out$biases <- .mlx_wrap_result(result$biases, device)
  } else {
    out$biases <- NULL
  }

  out
}

#' Dequantize a Matrix
#'
#' Reconstructs an approximate floating-point matrix from a quantized representation
#' produced by [mlx_quantize()].
#'
#' @param w An mlx array (the quantized weight matrix)
#' @param scales An mlx array (the quantization scales)
#' @param biases An optional mlx array (the quantization biases for affine mode). Default: NULL
#' @param group_size The group size used during quantization. Default: 64
#' @param bits The number of bits used during quantization. Default: 4
#' @param mode The quantization mode used: "affine" or "mxfp4". Default: "affine"
#' @inheritParams common_params
#'
#' @return An mlx array with the dequantized (approximate) floating-point weights
#'
#' @details
#' Dequantization unpacks the low-precision quantized weights and applies the scales
#' (and biases) to reconstruct approximate floating-point values. Note that some
#' precision is lost during quantization and cannot be recovered.
#'
#' @examples
#' w <- mlx_rand_normal(c(64, 32))
#' quant <- mlx_quantize(w, group_size = 32)
#' w_reconstructed <- mlx_dequantize(quant$w_q, quant$scales, quant$biases, group_size = 32)
#'
#' @seealso [mlx_quantize()], [mlx_quantized_matmul()]
#' @export
mlx_dequantize <- function(w, scales, biases = NULL, group_size = 64L, bits = 4L,
                            mode = "affine", device = mlx_default_device()) {
  w <- as_mlx(w)
  scales <- as_mlx(scales)

  biases_ptr <- NULL
  if (!is.null(biases)) {
    biases <- as_mlx(biases)
    biases_ptr <- biases$ptr
  }

  ptr <- cpp_mlx_dequantize(w$ptr, scales$ptr, biases_ptr,
                            as.integer(group_size), as.integer(bits), mode, device)
  .mlx_wrap_result(ptr, device)
}

#' Quantized Matrix Multiplication
#'
#' Performs matrix multiplication with a quantized weight matrix. This operation
#' is essential for efficient inference with quantized models, significantly reducing
#' memory usage and computation time while maintaining reasonable accuracy.
#'
#' @inheritParams mlx_array_required
#' @param w An mlx array. Either:
#'   \itemize{
#'     \item A quantized weight matrix (uint32) from [mlx_quantize()], or
#'     \item An unquantized weight matrix that will be quantized automatically
#'   }
#' @param scales An optional mlx array (the quantization scales). If NULL and w is
#'   unquantized, w will be quantized automatically. Default: NULL
#' @param biases An optional mlx array (biases to add). For affine quantization, this
#'   should be the quantization biases if w is pre-quantized. Default: NULL
#' @param transpose Whether to transpose the weight matrix. Default: TRUE
#' @param group_size The group size for quantization. Default: 64
#' @param bits The number of bits for quantization (typically 4 or 8). Default: 4
#' @param mode The quantization mode, either "affine" or "mxfp4". Default: "affine"
#' @inheritParams common_params
#'
#' @return An mlx array with the result of the quantized matrix multiplication
#'
#' @details
#' Quantized matrix multiplication uses low-precision representations (typically 4-bit or
#' 8-bit integers) for weights, which reduces memory footprint by up to 8x compared to
#' float32. The scales parameter contains the dequantization factors needed to reconstruct
#' approximate float values during computation.
#'
#' The group_size parameter controls the granularity of quantization - smaller groups
#' provide better accuracy but slightly higher memory usage.
#'
#' **Automatic Quantization**: If only w is provided (without scales), the function will
#' automatically quantize w using [mlx_quantize()] before performing the multiplication.
#' For repeated operations, it's more efficient to pre-quantize weights once using
#' [mlx_quantize()] and reuse them.
#'
#' @examples
#' # Automatic quantization (convenient but slower for repeated use)
#' x <- mlx_rand_normal(c(4, 64))
#' w <- mlx_rand_normal(c(128, 64))
#' result <- mlx_quantized_matmul(x, w, group_size = 32)
#'
#' # Pre-quantized weights (faster for repeated operations)
#' quant <- mlx_quantize(w, group_size = 32, bits = 4)
#' result <- mlx_quantized_matmul(x, quant$w_q, quant$scales, quant$biases, group_size = 32)
#'
#' @seealso [mlx_quantize()], [mlx_dequantize()], [mlx_gather_qmm()]
#' @export
mlx_quantized_matmul <- function(x, w, scales = NULL, biases = NULL, transpose = TRUE,
                                  group_size = 64L, bits = 4L, mode = "affine",
                                  device = mlx_default_device()) {
  x <- as_mlx(x)
  w <- as_mlx(w)

  # Auto-quantize if scales not provided
  if (is.null(scales)) {
    quant <- mlx_quantize(w, group_size, bits, mode, device)
    w <- quant$w_q
    scales <- quant$scales
    if (is.null(biases) && !is.null(quant$biases)) {
      biases <- quant$biases
    }
  } else {
    scales <- as_mlx(scales)
  }

  biases_ptr <- NULL
  if (!is.null(biases)) {
    biases <- as_mlx(biases)
    biases_ptr <- biases$ptr
  }

  ptr <- cpp_mlx_quantized_matmul(x$ptr, w$ptr, scales$ptr, biases_ptr,
                                   transpose, as.integer(group_size),
                                   as.integer(bits), mode, device)
  .mlx_wrap_result(ptr, device)
}

#' Gather-based Quantized Matrix Multiplication
#'
#' Performs quantized matrix multiplication with optional gather operations on inputs.
#' This is useful for combining embedding lookups with quantized linear transformations,
#' a common pattern in transformer models.
#'
#' @inheritParams mlx_array_required
#' @param w An mlx array (the quantized weight matrix)
#' @param scales An mlx array (the quantization scales)
#' @param biases An optional mlx array (biases to add). Default: NULL
#' @param lhs_indices An optional mlx array (indices for gathering from x). Default: NULL
#' @param rhs_indices An optional mlx array (indices for gathering from w). Default: NULL
#' @param transpose Whether to transpose the weight matrix. Default: TRUE
#' @param group_size The group size for quantization. Default: 64
#' @param bits The number of bits for quantization (typically 4 or 8). Default: 4
#' @param mode The quantization mode, either "affine" or "mxfp4". Default: "affine"
#' @param sorted_indices Whether the indices are sorted (enables optimizations). Default: FALSE
#' @inheritParams common_params
#'
#' @return An mlx array with the result of the gather-based quantized matrix multiplication
#'
#' @details
#' This function combines gather operations (indexed lookups) with quantized matrix
#' multiplication. When lhs_indices is provided, it performs `x[lhs_indices]` before
#' the multiplication. Similarly, rhs_indices gathers from the weight matrix.
#'
#' This is particularly efficient for transformer models where you need to look up
#' token embeddings and then apply a quantized linear transformation in one fused
#' operation.
#'
#' @seealso [mlx_quantized_matmul()]
#' @seealso [mlx.nn](https://ml-explore.github.io/mlx/build/html/python/nn.html)
#' @export
mlx_gather_qmm <- function(x, w, scales, biases = NULL, lhs_indices = NULL,
                            rhs_indices = NULL, transpose = TRUE, group_size = 64L,
                            bits = 4L, mode = "affine", sorted_indices = FALSE,
                            device = mlx_default_device()) {
  x <- as_mlx(x)
  w <- as_mlx(w)
  scales <- as_mlx(scales)

  biases_ptr <- NULL
  if (!is.null(biases)) {
    biases <- as_mlx(biases)
    biases_ptr <- biases$ptr
  }

  lhs_indices_ptr <- NULL
  if (!is.null(lhs_indices)) {
    lhs_indices <- as_mlx(lhs_indices)
    lhs_indices_ptr <- lhs_indices$ptr
  }

  rhs_indices_ptr <- NULL
  if (!is.null(rhs_indices)) {
    rhs_indices <- as_mlx(rhs_indices)
    rhs_indices_ptr <- rhs_indices$ptr
  }

  ptr <- cpp_mlx_gather_qmm(x$ptr, w$ptr, scales$ptr, biases_ptr,
                            lhs_indices_ptr, rhs_indices_ptr, transpose,
                            as.integer(group_size), as.integer(bits), mode,
                            sorted_indices, device)
  .mlx_wrap_result(ptr, device)
}
