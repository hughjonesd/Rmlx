#' Create a learnable linear transformation
#'
#' @param in_features Number of input features.
#' @param out_features Number of output features.
#' @param bias Should a bias term be included?
#' @param device Device for the parameters.
#' @return An object of class `mlx_module`.
#' @export
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
#' @export
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
#' @export
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
#' @export
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
#' @export
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

mlx_param <- function(env, name) {
  structure(
    list(env = env, name = name),
    class = "mlx_param"
  )
}

mlx_param_get <- function(param) {
  stopifnot(inherits(param, "mlx_param"))
  param$env[[param$name]]
}

mlx_param_set <- function(param, value) {
  stopifnot(inherits(param, "mlx_param"), is.mlx(value))
  param$env[[param$name]] <- value
  invisible(NULL)
}

#' Retrieve parameter tensors
#'
#' @param params A list of `mlx_param`.
#' @return List of `mlx` tensors.
#' @export
mlx_param_values <- function(params) {
  lapply(params, mlx_param_get)
}

#' Assign tensors back to parameters
#'
#' @param params A list of `mlx_param`.
#' @param values A list of tensors.
#' @export
mlx_param_set_values <- function(params, values) {
  stopifnot(length(params) == length(values))
  for (i in seq_along(params)) {
    mlx_param_set(params[[i]], values[[i]])
  }
  invisible(NULL)
}
