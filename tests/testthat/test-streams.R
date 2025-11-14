test_that("streams can be created and set", {
  dev <- mlx_default_device()
  stream <- mlx_new_stream(dev)
  expect_s3_class(stream, "mlx_stream")
  expect_equal(stream$device, dev)

  old <- mlx_default_stream(dev)
  on.exit(mlx_set_default_stream(old), add = TRUE)

  mlx_set_default_stream(stream)
  current <- mlx_default_stream(dev)
  expect_equal(current$index, stream$index)
})

test_that("operations accept stream arguments", {
  dev <- mlx_default_device()
  stream <- mlx_new_stream(dev)

  zeros <- mlx_zeros(c(2, 2), device = stream)
  expect_equal(zeros$device, dev)
  expect_equal(as.matrix(zeros), matrix(0, 2, 2))

  ones <- mlx_ones(c(2, 2), device = stream)
  expect_equal(as.matrix(ones), matrix(1, 2, 2))

  rnd <- mlx_rand_uniform(c(3, 3), device = stream)
  expect_equal(rnd$device, dev)

  perm <- mlx_rand_permutation(5, device = stream)
  expect_equal(perm$device, dev)
  expect_equal(sort(as.vector(perm)), 1:5)
})

test_that("mlx_synchronize accepts stream", {
  dev <- mlx_default_device()
  stream <- mlx_new_stream(dev)
  expect_silent(mlx_synchronize(stream))
})
