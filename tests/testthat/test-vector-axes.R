test_that("vector sort accepts an explicit axis and preserves names", {
  values <- setNames(c(4, 1, 3, 2), letters[1:4])
  x <- as_mlx(values)
  expect_null(dim(x))
  expect_equal(mlx_shape(x), 4L)
  sorted <- mlx_sort(x, axis = 1L)
  expect_equal(as.vector(sorted), unname(sort(values)))
  expect_equal(names(sorted), names(sort(values)))
  expect_error(mlx_sort(x, axis = 2L), "out of bounds")
  expect_error(mlx_sort(mlx_scalar(1), axis = 1L), "out of bounds")
})

test_that("vector argsort accepts an explicit axis", {
  values <- c(4, 1, 3, 2)
  expect_equal(as.integer(mlx_argsort(values, axis = 1L)), order(values))
})

test_that("vector argmax accepts an explicit axis", {
  expect_equal(as.integer(mlx_argmax(c(4, 1, 3, 2), axis = 1L)), 1L)
  expect_equal(mlx_shape(mlx_argmax(1:4, axis = 1L, drop = FALSE)), 1L)
})

test_that("vector argmin accepts an explicit axis", {
  expect_equal(as.integer(mlx_argmin(c(4, 1, 3, 2), axis = 1L)), 2L)
})

test_that("vector top-k accepts an explicit axis", {
  expect_equal(sort(as.vector(mlx_topk(c(4, 1, 3, 2), k = 2L, axis = 1L))), c(3, 4))
})

test_that("vector partition accepts an explicit axis", {
  part <- as.vector(mlx_partition(c(4, 1, 3, 2), kth = 1L, axis = 1L))
  expect_equal(part[2L], 2)
  expect_equal(sort(part), 1:4)
})

test_that("vector argpartition accepts an explicit axis", {
  values <- c(4, 1, 3, 2)
  indices <- as.integer(mlx_argpartition(values, kth = 1L, axis = 1L))
  expect_equal(values[indices][2L], 2)
  expect_equal(sort(indices), 1:4)
})

test_that("vector logsumexp accepts an explicit axis", {
  values <- c(4, 1, 3, 2)
  expect_equal(as.numeric(mlx_logsumexp(values, axes = 1L)), log(sum(exp(values))),
               tolerance = 1e-6)
})

test_that("vector logcumsumexp accepts an explicit axis", {
  values <- c(4, 1, 3, 2)
  expect_equal(as.vector(mlx_logcumsumexp(values, axis = 1L)), log(cumsum(exp(values))),
               tolerance = 1e-6)
})

test_that("vector softmax accepts an explicit axis", {
  values <- c(4, 1, 3, 2)
  expect_equal(as.vector(mlx_softmax(values, axes = 1L)), exp(values) / sum(exp(values)),
               tolerance = 1e-6)
})

test_that("mlx vectors can stack along the second insertion axis", {
  x <- as_mlx(1:4)
  y <- as_mlx(5:8)
  expect_equal(as.matrix(mlx_stack(x, y, axis = 2L)), cbind(1:4, 5:8))
  expect_equal(as.matrix(mlx_stack(x, y, axis = 1L)), rbind(1:4, 5:8))
  expect_error(mlx_stack(x, y, axis = 3L), "axes must be between")
})

test_that("mlx vectors split into equal sections using their native axis length", {
  x <- as_mlx(setNames(1:4, letters[1:4]))
  parts <- mlx_split(x, sections = 2L, axis = 1L)
  expect_equal(lapply(parts, as.vector), list(1:2, 3:4))
  expect_equal(lapply(parts, names), list(letters[1:2], letters[3:4]))
})

test_that("mlx vectors split at explicit positions using their native axis length", {
  parts <- mlx_split(1:4, sections = list(1L, 3L), axis = 1L)
  expect_equal(lapply(parts, as.vector), list(1, 2:3, 4))
})

test_that("asplit accepts the vector margin", {
  x <- as_mlx(setNames(1:4, letters[1:4]))
  expect_equal(unname(lapply(asplit(x, MARGIN = 1L), as.vector)), as.list(1:4))
})

test_that("vector axis movement preserves values and names", {
  x <- as_mlx(setNames(1:4, letters[1:4]))
  moved <- mlx_moveaxis(x, source = 1L, destination = 1L)
  expect_equal(as.vector(moved), 1:4)
  expect_equal(names(moved), letters[1:4])
})

test_that("vector aperm validates the permutation", {
  x <- as_mlx(1:4)
  expect_error(aperm(x, 2L), "perm must be a permutation")
  expect_equal(as.vector(aperm(x, 1L)), 1:4)
})

test_that("vector flatten validates its axis range", {
  x <- as_mlx(1:4)
  expect_error(mlx_flatten(x, start_axis = 2L), "out of bounds")
  expect_error(mlx_flatten(x, end_axis = 2L), "out of bounds")
  expect_equal(as.vector(mlx_flatten(x)), 1:4)
  expect_equal(as.vector(mlx_flatten(x, start_axis = 1L, end_axis = 1L)), 1:4)
})

test_that("binding along the vector axis uses its native length", {
  expect_equal(as.vector(abind(as_mlx(1:2), as_mlx(3:4), along = 1L)), 1:4)
})

test_that("categorical sampling infers the axis of vector logits", {
  logits <- as_mlx(c(-Inf, 0, -Inf))
  expect_equal(as.integer(mlx_rand_categorical(logits, num_samples = 5L)), rep(2L, 5))
})

test_that("categorical sampling accepts an explicit vector axis", {
  logits <- as_mlx(c(-Inf, 0, -Inf))
  expect_equal(as.integer(mlx_rand_categorical(logits, axis = 1L, num_samples = 5L)),
               rep(2L, 5))
})

test_that("softmax layers infer the axis of vectors", {
  values <- 1:4
  expected <- exp(values) / sum(exp(values))
  expect_equal(as.vector(mlx_forward(mlx_softmax_layer(), as_mlx(values))), expected,
               tolerance = 1e-6)
  expect_equal(as.vector(mlx_forward(mlx_softmax_layer(), values)), expected,
               tolerance = 1e-6)
})

test_that("layer normalization uses the single dimension of vectors", {
  values <- 1:4
  eps <- 1e-5
  expected <- (values - mean(values)) / sqrt(mean((values - mean(values))^2) + eps)
  expect_equal(as.vector(mlx_forward(mlx_layer_norm(4L, eps = eps), as_mlx(values))),
               expected, tolerance = 1e-6)
  expect_equal(as.vector(mlx_forward(mlx_layer_norm(4L, eps = eps), values)), expected,
               tolerance = 1e-6)
})

test_that("dropout requests one mask value per vector element", {
  # Use a fixed mask so the regression does not depend on random draws.
  testthat::local_mocked_bindings(mlx_rand_bernoulli = function(dim, prob) {
    expect_equal(dim, 4L)
    expect_equal(prob, 0.5)
    as_mlx(c(0, 1, 0, 1))
  }, .package = "Rmlx")
  dropped <- mlx_forward(mlx_dropout(p = 0.5), mlx_ones(4L))
  expect_equal(mlx_shape(dropped), 4L)
  expect_equal(as.vector(dropped), c(0, 2, 0, 2))
})

test_that("vector cumulative sums accept an explicit axis", {
  expect_equal(as.vector(mlx_cumsum(1:4, axis = 1L)), cumsum(1:4))
})

test_that("vector cumulative products accept an explicit axis", {
  expect_equal(as.vector(mlx_cumprod(1:4, axis = 1L)), cumprod(1:4))
})

test_that("vector quantiles accept an explicit axis", {
  probs <- c(0.25, 0.5, 0.75)
  expect_equal(as.vector(mlx_quantile(1:4, probs, axis = 1L)),
               unname(stats::quantile(1:4, probs)), tolerance = 1e-6)
})

test_that("singleton vector quantiles broadcast using their native shape", {
  out <- mlx_quantile(7, c(0, 0.5, 1), axis = 1L)
  expect_equal(mlx_shape(out), 3L)
  expect_equal(as.vector(out), rep(7, 3))
})

test_that("vector squeeze accepts an explicit singleton axis", {
  expect_equal(mlx_shape(mlx_squeeze(as_mlx(7), axes = 1L)), integer(0L))
})

test_that("vector repeat accepts an explicit axis", {
  expect_equal(as.vector(mlx_repeat(1:4, repeats = 2L, axis = 1L)), rep(1:4, each = 2L))
})

test_that("vector roll accepts an explicit axis", {
  expect_equal(as.vector(mlx_roll(1:4, shift = 1L, axes = 1L)), c(4, 1, 2, 3))
})

test_that("vector padding accepts an explicit axis", {
  expect_equal(as.vector(mlx_pad(1:4, pad_width = 1L, axes = 1L)), c(0, 1:4, 0))
})

test_that("vector swapaxes accepts its single axis", {
  expect_equal(as.vector(mlx_swapaxes(1:4, axis1 = 1L, axis2 = 1L)), 1:4)
})

test_that("vector take along axis accepts its single axis", {
  expect_equal(as.vector(mlx_take_along_axis(1:4, c(4L, 2L), axis = 1L)), c(4, 2))
})

test_that("vector put along axis accepts its single axis", {
  expect_equal(as.vector(mlx_put_along_axis(1:4, c(4L, 2L), c(10, 20), axis = 1L)),
               c(1, 20, 3, 10))
})

test_that("vector scatter add accepts its single axis", {
  expect_equal(as.vector(mlx_scatter_add_axis(1:4, c(4L, 2L), c(10, 20), axis = 1L)),
               c(1, 22, 3, 14))
})

test_that("random permutation accepts a vector axis", {
  expect_equal(sort(as.vector(mlx_rand_permutation(as_mlx(1:4), axis = 1L))), 1:4)
})

test_that("vector FFT accepts an explicit axis", {
  expect_equal(as.vector(mlx_fft(1:4, axis = 1L)), stats::fft(1:4), tolerance = 1e-6)
})

test_that("vector FFTn accepts explicit axes", {
  expect_equal(as.vector(mlx_fftn(1:4, axes = 1L)), stats::fft(1:4), tolerance = 1e-6)
})

test_that("vector FFT2 accepts an explicit single axis", {
  expect_equal(as.vector(mlx_fft2(1:4, axes = 1L)), stats::fft(1:4), tolerance = 1e-6)
})
