test_that("embedding lookup preserves token layout and scalar output shape", {
  weights <- matrix(seq_len(15), 5, 3)
  token_inputs <- list(
    3L,
    c(4L, 2L, 4L),
    matrix(c(4L, 1L, 2L, 4L), 2, 2),
    array(c(5L, 1L, 3L, 2L), c(2, 1, 2))
  )

  for (device in c("cpu", "gpu")) {
    with_device(device, {
      embedding <- mlx_embedding(5L, 3L)
      mlx_param_set_values(mlx_parameters(embedding), list(as_mlx(weights)))
      for (tokens in token_inputs) {
        token_shape <- if (is.null(dim(tokens))) length(tokens) else dim(tokens)
        expected <- array(weights[as.integer(tokens), , drop = FALSE],
                          dim = c(token_shape, 3L))
        for (input in list(tokens, as_mlx(tokens), as_mlx(tokens, dtype = "int32"))) {
          output <- mlx_forward(embedding, input)
          expect_equal(as.array(output), expected, tolerance = 1e-6)
          expect_equal(mlx_shape(output), as.integer(c(token_shape, 3L)))
        }
      }
      scalar_output <- mlx_forward(embedding, mlx_scalar(3L, dtype = "int32"))
      expect_equal(mlx_shape(scalar_output), c(1L, 3L))
      expect_equal(as.matrix(scalar_output), weights[3L, , drop = FALSE])
    })
  }
})

test_that("embedding gradients accumulate repeated tokens and leave unused rows zero", {
  weights <- matrix(seq_len(15) / 10, 5, 3)
  token_inputs <- list(
    c(4L, 2L, 4L),
    matrix(c(4L, 2L, 4L, 1L), 2, 2),
    array(c(4L, 2L, 4L, 1L), c(2, 1, 2))
  )

  for (device in c("cpu", "gpu")) {
    with_device(device, {
      embedding <- mlx_embedding(5L, 3L)
      params <- mlx_parameters(embedding)
      weight <- as_mlx(weights)
      for (tokens in token_inputs) {
        expected <- 2 * weights * tabulate(as.integer(tokens), nbins = 5L)
        loss <- function(w) {
          mlx_param_set_values(params, list(w))
          sum(mlx_forward(embedding, tokens)^2)
        }
        gradient <- mlx_grad(loss, weight)[[1L]]
        expect_equal(as.matrix(gradient), expected, tolerance = 1e-6)
        expect_equal(mlx_shape(gradient), c(5L, 3L))
      }
      tokens <- mlx_scalar(4L, dtype = "int32")
      expected <- matrix(0, 5, 3)
      expected[4L, ] <- 2 * weights[4L, ]
      gradient <- mlx_grad(loss, weight)[[1L]]
      expect_equal(as.matrix(gradient), expected, tolerance = 1e-6)
    })
  }
})

test_that("embedding lookup preserves float64 weights on CPU", {
  with_device("cpu", {
    weights <- matrix(c(1 + 2^-30, 2 + 2^-30, 3 + 2^-30, 4 + 2^-30), 2, 2)
    embedding <- mlx_embedding(2L, 2L)
    mlx_param_set_values(mlx_parameters(embedding), list(as_mlx(weights, dtype = "float64")))
    output <- mlx_forward(embedding, c(2L, 1L))
    expect_equal(mlx_dtype(output), "float64")
    expect_equal(as.matrix(output), weights[c(2L, 1L), , drop = FALSE], tolerance = 1e-12)
  })
})

test_that("embedding lookup rejects token IDs outside the vocabulary", {
  for (device in c("cpu", "gpu")) {
    with_device(device, {
      embedding <- mlx_embedding(5L, 3L)
      for (tokens in list(c(1L, 0L), c(1L, 6L), c(1L, -1L))) {
        expect_error(mlx_forward(embedding, tokens), "out of range")
        expect_error(mlx_forward(embedding, as_mlx(tokens, dtype = "int32")), "out of range")
      }
    })
  }
})

test_that("embedding lookup rejects wide integer IDs without wrapping", {
  for (device in c("cpu", "gpu")) {
    with_device(device, {
      embedding <- mlx_embedding(5L, 3L)
      tokens <- as_mlx(c(1, 2^32 + 1), dtype = "uint64")
      expect_error(mlx_forward(embedding, tokens), "out of range")
    })
  }
})
