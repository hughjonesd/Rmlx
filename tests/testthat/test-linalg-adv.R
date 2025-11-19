test_that("mlx_norm matches base computations", {
  mat <- matrix(c(1, -2, 3, 4), 2, 2)
  x <- as_mlx(mat)

  frob <- as.numeric(mlx_norm(x))
  expect_equal(frob, base::norm(mat, type = "F"), tolerance = 1e-6)

  inf_norm <- as.numeric(mlx_norm(x, ord = Inf))
  expect_equal(inf_norm, base::norm(mat, type = "I"), tolerance = 1e-6)

  row_norms <- as.numeric(mlx_norm(x, axes = 2))
  expected_rows <- apply(mat, 1, function(row) sqrt(sum(row^2)))
  expect_equal(row_norms, expected_rows, tolerance = 1e-6)
})

test_that("mlx_eig recreates eigen decomposition", {
  mat <- matrix(c(2, -1, 0, 2), 2, 2)
  x <- as_mlx(mat)
  eig_res <- mlx_eig(x)

  values <- as.vector(eig_res$values)
  vectors <- as.matrix(eig_res$vectors)

  recon <- mat %*% vectors
  expected <- vectors %*% diag(values)
  expect_equal(recon, expected, tolerance = 1e-5)
})

test_that("mlx_eigvals matches base eigenvalues", {
  mat <- matrix(c(3, 1, 0, 2), 2, 2)
  vals <- sort(Re(as.vector(mlx_eigvals(as_mlx(mat)))))
  expected <- sort(base::eigen(mat, only.values = TRUE)$values)
  expect_equal(vals, expected, tolerance = 1e-6)
})

test_that("mlx_eigh and mlx_eigvalsh agree with symmetric eigenvalues", {
  mat <- matrix(c(2, 1, 1,
                  1, 3, 0,
                  1, 0, 4), 3, 3, byrow = TRUE)
  x <- as_mlx(mat)

  base_eig <- eigen(mat, symmetric = TRUE)
  vals <- as.vector(mlx_eigvalsh(x))
  expect_equal(sort(vals), sort(base_eig$values), tolerance = 1e-6)

  eigh <- mlx_eigh(x)
  vecs <- as.matrix(eigh$vectors)
  check <- mat %*% vecs
  expect_equal(check, vecs %*% diag(as.vector(eigh$values)),
               tolerance = 1e-5)
})

test_that("mlx_solve_triangular matches base solve", {
  lower <- matrix(c(2, 0, 1, 3), 2, 2, byrow = TRUE)
  rhs <- matrix(c(1, 5), 2, 1)

  sol_lower <- as.matrix(mlx_solve_triangular(as_mlx(lower), as_mlx(rhs), upper = FALSE))
  expect_equal(sol_lower, forwardsolve(lower, rhs), tolerance = 1e-6)

  upper <- matrix(c(3, 2, 0, 4), 2, 2, byrow = TRUE)
  rhs2 <- matrix(c(4, 2), 2, 1)
  sol_upper <- as.matrix(mlx_solve_triangular(as_mlx(upper), as_mlx(rhs2), upper = TRUE))
  expect_equal(sol_upper, backsolve(upper, rhs2), tolerance = 1e-6)
})

test_that("mlx_cross matches manual cross product", {
  a <- matrix(c(1, 0, 0,
                0, 1, 0), nrow = 2, byrow = TRUE)
  b <- matrix(c(0, 1, 0,
                0, 0, 1), nrow = 2, byrow = TRUE)
  cross_res <- as.matrix(mlx_cross(as_mlx(a), as_mlx(b), axis = 2))

  expected <- t(apply(cbind(a, b), 1, function(row) {
    u <- row[1:3]
    v <- row[4:6]
    c(u[2] * v[3] - u[3] * v[2],
      u[3] * v[1] - u[1] * v[3],
      u[1] * v[2] - u[2] * v[1])
  }))
  expect_equal(cross_res, expected, tolerance = 1e-6)
})
