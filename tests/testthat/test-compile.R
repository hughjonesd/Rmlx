test_that("basic compilation works", {
  matmul_add <- function(x, w, b) {
    (x %*% w) + b
  }

  compiled_fn <- mlx_compile(matmul_add)

  x <- mlx_rand_normal(c(10, 64))
  w <- mlx_rand_normal(c(64, 128))
  b <- mlx_rand_normal(c(128))

  result <- compiled_fn(x, w, b)

  expect_s3_class(result, "mlx")
  expect_equal(mlx_shape(result), c(10, 128))
})

test_that("compiled functions can be called multiple times", {
  square_sum <- function(x) {
    sum(x * x)
  }

  compiled_fn <- mlx_compile(square_sum)

  x1 <- mlx_rand_normal(c(5, 5))
  x2 <- mlx_rand_normal(c(5, 5))

  r1 <- compiled_fn(x1)
  r2 <- compiled_fn(x2)

  expect_s3_class(r1, "mlx")
  expect_s3_class(r2, "mlx")

  # Results should be different for different inputs
  expect_false(isTRUE(all.equal(as.matrix(r1), as.matrix(r2))))
})

test_that("compiled functions handle multiple returns", {
  forward_and_norm <- function(x, w) {
    y <- x %*% w
    norm_val <- sqrt(sum(y * y))
    list(y, norm_val)
  }

  compiled_fn <- mlx_compile(forward_and_norm)

  x <- mlx_rand_normal(c(10, 64))
  w <- mlx_rand_normal(c(64, 128))

  result <- compiled_fn(x, w)

  expect_type(result, "list")
  expect_length(result, 2)
  expect_s3_class(result[[1]], "mlx")
  expect_s3_class(result[[2]], "mlx")
  expect_equal(mlx_shape(result[[1]]), c(10, 128))
  expect_equal(mlx_shape(result[[2]]), integer(0))  # Scalar
})

test_that("shapeless compilation works", {
  square <- function(x) x * x

  # Shapeless compilation
  compiled_fn <- mlx_compile(square, shapeless = TRUE)

  # First call with shape (5, 5)
  x1 <- mlx_rand_normal(c(5, 5))
  r1 <- compiled_fn(x1)
  expect_equal(mlx_shape(r1), c(5, 5))

  # Second call with different shape (10, 10) - should work without recompilation
  x2 <- mlx_rand_normal(c(10, 10))
  r2 <- compiled_fn(x2)
  expect_equal(mlx_shape(r2), c(10, 10))

  # Third call with 1D array
  x3 <- mlx_rand_normal(c(20))
  r3 <- compiled_fn(x3)
  expect_equal(mlx_shape(r3), 20)
})

test_that("compilation with complex operations works", {
  complex_fn <- function(x, y) {
    a <- x + y
    b <- x - y
    c <- a * b
    d <- sqrt(abs(c))
    mean(d)
  }

  compiled_fn <- mlx_compile(complex_fn)

  x <- mlx_rand_normal(c(20, 20))
  y <- mlx_rand_normal(c(20, 20))

  result <- compiled_fn(x, y)

  expect_s3_class(result, "mlx")
  expect_equal(length(mlx_shape(result)), 0)  # Scalar
})

test_that("mlx_disable_compile and mlx_enable_compile work", {
  simple_fn <- function(x) x * 2

  compiled_fn <- mlx_compile(simple_fn)

  x <- mlx_rand_normal(c(5, 5))

  # Should work with compilation enabled
  r1 <- compiled_fn(x)
  expect_s3_class(r1, "mlx")

  # Disable compilation
  mlx_disable_compile()

  # Should still work (but without optimization)
  r2 <- compiled_fn(x)
  expect_s3_class(r2, "mlx")

  # Re-enable
  mlx_enable_compile()

  # Should work again
  r3 <- compiled_fn(x)
  expect_s3_class(r3, "mlx")
})

test_that("compilation fails with informative error for non-mlx returns", {
  bad_fn <- function(x) {
    42  # Returns numeric, not mlx
  }

  compiled_fn <- mlx_compile(bad_fn)
  x <- mlx_rand_normal(c(5, 5))

  expect_error(
    compiled_fn(x),
    "Compiled function must return either"
  )
})

test_that("compiled functions auto-convert arguments to mlx", {
  add_fn <- function(x, y) x + y

  compiled_fn <- mlx_compile(add_fn)

  # Pass R matrices - should auto-convert
  x_r <- matrix(1:6, 2, 3)
  y_r <- matrix(7:12, 2, 3)

  result <- compiled_fn(x_r, y_r)

  expect_s3_class(result, "mlx")
  expect_equal(mlx_shape(result), c(2, 3))
})

test_that("compiled functions preserve list names", {
  # Simple function returning a named list
  test_func <- function(x, y) {
    sum_result <- x + y
    product_result <- x * y
    list(sum = sum_result, product = product_result)
  }

  x <- as_mlx(matrix(1:5, ncol = 1))
  y <- as_mlx(matrix(6:10, ncol = 1))

  # Uncompiled: should have names
  result_uncompiled <- test_func(x, y)
  expect_equal(names(result_uncompiled), c("sum", "product"))
  expect_s3_class(result_uncompiled$sum, "mlx")
  expect_s3_class(result_uncompiled$product, "mlx")

  # Compiled: should also preserve names
  test_func_compiled <- mlx_compile(test_func)
  result_compiled <- test_func_compiled(x, y)

  expect_type(result_compiled, "list")
  expect_length(result_compiled, 2)
  expect_equal(names(result_compiled), c("sum", "product"))
  expect_s3_class(result_compiled$sum, "mlx")
  expect_s3_class(result_compiled$product, "mlx")

  # Verify we can access by name
  expect_equal(mlx_shape(result_compiled$sum), c(5, 1))
  expect_equal(mlx_shape(result_compiled$product), c(5, 1))
})
