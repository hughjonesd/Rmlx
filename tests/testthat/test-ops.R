test_that("arithmetic operations work", {
  x <- matrix(1:12, 3, 4)
  y <- matrix(13:24, 3, 4)

  x_mlx <- as_mlx(x)
  y_mlx <- as_mlx(y)

  # Addition
  z <- as.matrix(x_mlx + y_mlx)
  expect_equal(z, x + y, tolerance = 1e-6)

  # Subtraction
  z <- as.matrix(x_mlx - y_mlx)
  expect_equal(z, x - y, tolerance = 1e-6)

  # Multiplication
  z <- as.matrix(x_mlx * y_mlx)
  expect_equal(z, x * y, tolerance = 1e-6)

  # Division
  z <- as.matrix(x_mlx / y_mlx)
  expect_equal(z, x / y, tolerance = 1e-6)

  # Power
  z <- as.matrix(x_mlx ^ 2)
  expect_equal(z, x ^ 2, tolerance = 1e-6)
})

test_that("unary negation works", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  z <- as.matrix(-x_mlx)
  expect_equal(z, -x, tolerance = 1e-6)
})

test_that("comparison operators work", {
  x <- matrix(1:12, 3, 4)
  y <- matrix(c(1:6, 13:18), 3, 4)

  x_mlx <- as_mlx(x)
  y_mlx <- as_mlx(y)

  # Less than
  lt <- x_mlx < y_mlx
  z <- as.matrix(lt)
  expect_equal(z, x < y)
  expect_equal(mlx_dtype(lt), "bool")

  # Equal
  z <- as.matrix(x_mlx == y_mlx)
  expect_equal(z, x == y)

  # Greater than
  z <- as.matrix(x_mlx > y_mlx)
  expect_equal(z, x > y)
})

test_that("scalar operations work", {
  x <- matrix(1:12, 3, 4)
  x_mlx <- as_mlx(x)

  # Scalar addition
  z <- as.matrix(x_mlx + 10)
  expect_equal(z, x + 10, tolerance = 1e-6)

  # Scalar multiplication
  z <- as.matrix(x_mlx * 2)
  expect_equal(z, x * 2, tolerance = 1e-6)
})

test_that("boolean operands coerce for arithmetic", {
  bool_mat <- matrix(c(TRUE, FALSE, TRUE, FALSE), 2, 2)
  num_mat <- matrix(1:4, 2, 2)

  bool_mlx <- as_mlx(bool_mat)
  num_mlx <- as_mlx(num_mat)

  sum_obj <- bool_mlx + num_mlx
  sum_res <- as.matrix(sum_obj)
  expect_equal(sum_res, num_mat + (bool_mat * 1), tolerance = 1e-6)
  expect_equal(mlx_dtype(sum_obj), "float32")

  bool_sum_obj <- bool_mlx + bool_mlx
  bool_sum <- as.matrix(bool_sum_obj)
  expect_equal(bool_sum, (bool_mat * 1) + (bool_mat * 1), tolerance = 1e-6)
  expect_equal(mlx_dtype(bool_sum_obj), "float32")
})

test_that("binary operations align devices and dtypes", {
  old_device <- mlx_default_device()
  on.exit(mlx_default_device(old_device))

  mlx_default_device("gpu")

  x_gpu <- as_mlx(matrix(1:4, 2, 2), device = "gpu", dtype = "float32")
  y_cpu <- as_mlx(matrix(5:8, 2, 2), device = "cpu")

  result <- x_gpu + y_cpu

  expect_equal(result$device, "gpu")
  expect_equal(mlx_dtype(result), "float32")
  expect_equal(as.matrix(result), matrix(c(6, 8, 10, 12), 2, 2), tolerance = 1e-6)
})

test_that("arithmetic works on non-contiguous views", {
  seed <- as.integer(format(Sys.Date(), "%Y%m%d"))
  set.seed(seed)
  base <- matrix(sample(-5:5, 12, replace = TRUE), 3, 4)
  x <- as_mlx(base)

  lhs <- x[c(3L, 1L), c(4L, 2L)]
  rhs <- x[c(2L, 3L), c(1L, 4L)]

  expect_equal(as.matrix(lhs + rhs),
               base[c(3, 1), c(4, 2)] + base[c(2, 3), c(1, 4)],
               tolerance = 1e-6)

  expect_equal(as.matrix(lhs * rhs),
               base[c(3, 1), c(4, 2)] * base[c(2, 3), c(1, 4)],
               tolerance = 1e-6)
})

test_that("logical operators work", {
  old_device <- mlx_default_device()
  on.exit(mlx_default_device(old_device))
  mlx_default_device("gpu")

  a <- matrix(c(TRUE, FALSE, TRUE, FALSE), 2, 2)
  b <- matrix(c(TRUE, TRUE, FALSE, FALSE), 2, 2)

  a_mlx <- as_mlx(a, device = "gpu")
  b_mlx <- as_mlx(b, device = "cpu")

  expect_equal(as.matrix(a_mlx & b_mlx), a & b)
  expect_equal(as.matrix(a_mlx | b_mlx), a | b)

  res_and <- a_mlx & b_mlx
  expect_equal(mlx_dtype(res_and), "bool")
  expect_equal(res_and$device, "gpu")

  # Unary not
  expect_equal(as.matrix(!a_mlx), !a)

  # Logical operators with scalar coercion
  expect_equal(as.matrix(a_mlx & TRUE), a & TRUE)
  expect_equal(as.matrix(a_mlx | 0), a | FALSE)

  # Short-circuit variants are not S3 generic; ensure evaluation still works via base fallback
  expect_true(isTRUE(as.vector(as.matrix(a_mlx))[1] && as.vector(as.matrix(b_mlx))[1]))
})

test_that("mlx_minimum and mlx_maximum compute elementwise extrema", {
  x <- matrix(c(-1, 2, 3, 4), 2, 2)
  y <- matrix(c(4, 1, 0, 5), 2, 2)

  min_res <- mlx_minimum(x, y)
  max_res <- mlx_maximum(as_mlx(x, dtype = "float32"), 1)

  expect_equal(as.matrix(min_res), pmin(x, y), tolerance = 1e-6)
  expect_equal(mlx_dtype(min_res), "float32")

  expect_equal(as.matrix(max_res), pmax(x, 1), tolerance = 1e-6)
  expect_equal(mlx_dtype(max_res), "float32")
})

test_that("mlx_clip clamps values", {
  x <- mlx_matrix(seq(-2, 2, length.out = 4), 2, 2)

  clipped <- mlx_clip(x, min = -1, max = 1)
  expect_equal(as.matrix(clipped), pmin(pmax(as.matrix(x), -1), 1), tolerance = 1e-6)

  clipped_upper <- mlx_clip(x, max = 0.5)
  expect_true(all(as.matrix(clipped_upper) <= 0.5 + 1e-6))

  clipped_lower <- mlx_clip(x, min = 0)
  expect_true(all(as.matrix(clipped_lower) >= -1e-6))

  expect_error(mlx_clip(x, min = 2, max = 1), "min must be less than or equal to max")
})

test_that("floor division and modulo work", {
  x <- mlx_matrix(c(5, -5, 10, -10), 2, 2)
  y <- mlx_matrix(c(2, 2, -3, -3), 2, 2)

  floor_res <- x %/% y
  mod_res <- x %% y

  expect_equal(as.matrix(floor_res), matrix(c(2, -3, -4, 3), 2, 2), tolerance = 1e-6)
  expect_equal(as.matrix(mod_res), matrix(c(1, 1, -2, -1), 2, 2), tolerance = 1e-6)
  expect_equal(mlx_dtype(floor_res), "float32")
  expect_equal(mlx_dtype(mod_res), "float32")
})
