library(Rmlx)
library(bench)

`%||%` <- function(x, y) if (!is.null(x)) x else y

force_mlx <- function(x) {
  if (inherits(x, "mlx")) {
    mlx_eval(x)
  } else if (is.list(x)) {
    lapply(x, force_mlx)
  }
  invisible(NULL)
}

encode_matrix32 <- function(mat) {
  list(
    dim = dim(mat),
    raw = writeBin(as.vector(mat), raw(), size = 4)
  )
}

decode_matrix32 <- function(obj) {
  vals <- readBin(obj$raw, what = "double", size = 4, n = prod(obj$dim))
  matrix(vals, nrow = obj$dim[1], ncol = obj$dim[2])
}

build_distribution_inputs <- function(sizes, seed = 20251031L) {
  make_payload <- function(base_data) {
    mlx_data <- list(
      vec = as_mlx(base_data$vec, dtype = "float32"),
      prob = as_mlx(base_data$prob, dtype = "float32")
    )
    force_mlx(mlx_data)
    list(base = base_data, mlx = mlx_data)
  }

  regenerate <- function(n) {
    set.seed(seed + n)
    vec <- rnorm(n)
    prob <- runif(n)

    base_data <- list(
      vec = vec,
      prob = prob
    )
    make_payload(base_data)
  }

  lapply(sizes, regenerate)
}

build_benchmark_inputs <- function(sizes, seed = 20251031L, cache_dir = NULL) {
  if (!is.null(cache_dir)) {
    dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
  }

  make_payload <- function(base_data) {
    mlx_data <- list(
      a = as_mlx(base_data$a, dtype = "float32"),
      b = as_mlx(base_data$b, dtype = "float32"),
      spd = as_mlx(base_data$spd, dtype = "float32"),
      rhs = as_mlx(base_data$rhs, dtype = "float32"),
      chol = as_mlx(base_data$chol, dtype = "float32"),
      idx_vec = base_data$idx_vec,
      idx_mat = base_data$idx_mat,
      vec = as_mlx(base_data$vec, dtype = "float32"),
      prob = as_mlx(base_data$prob, dtype = "float32")
    )
    force_mlx(mlx_data)
    list(base = base_data, mlx = mlx_data)
  }

  regenerate <- function(n) {
    cache_path <- if (is.null(cache_dir)) NULL else file.path(cache_dir, sprintf("inputs_%s.rds", n))
    set.seed(seed + n)
    a <- matrix(rnorm(n * n), n, n)
    b <- matrix(rnorm(n * n), n, n)
    spd <- crossprod(a) + diag(n) * 1e-3
    rhs <- matrix(rnorm(n), n, 1)
    chol_base <- chol(spd)
    idx_vec <- sample.int(n, size = n, replace = TRUE)
    idx_mat <- cbind(
      sample.int(n, size = n, replace = TRUE),
      sample.int(n, size = n, replace = TRUE)
    )
    vec <- rnorm(n)
    prob <- runif(n)

    base_data <- list(
      a = a,
      b = b,
      spd = spd,
      rhs = rhs,
      chol = chol_base,
      idx_vec = idx_vec,
      idx_mat = idx_mat,
      vec = vec,
      prob = prob
    )
    if (!is.null(cache_path)) {
      cache_obj <- list(
        a = encode_matrix32(a),
        b = encode_matrix32(b),
        rhs = encode_matrix32(rhs),
        idx_vec = idx_vec,
        idx_mat = idx_mat,
        vec = vec,
        prob = prob
      )
      saveRDS(cache_obj, cache_path, compress = "xz")
    }

    make_payload(base_data)
  }

  load_from_cache <- function(cache_path, n) {
    cache_obj <- readRDS(cache_path)
    a <- decode_matrix32(cache_obj$a)
    b <- decode_matrix32(cache_obj$b)
    rhs <- decode_matrix32(cache_obj$rhs)
    spd <- crossprod(a) + diag(n) * 1e-3
    chol_base <- chol(spd)

    # For backwards compatibility with old cache files
    if (is.null(cache_obj$vec) || is.null(cache_obj$prob)) {
      set.seed(20251031L + n + 1000L)
      vec <- rnorm(n)
      prob <- runif(n)
    } else {
      vec <- cache_obj$vec
      prob <- cache_obj$prob
    }

    base_data <- list(
      a = a,
      b = b,
      spd = spd,
      rhs = rhs,
      chol = chol_base,
      idx_vec = cache_obj$idx_vec,
      idx_mat = cache_obj$idx_mat,
      vec = vec,
      prob = prob
    )
    make_payload(base_data)
  }

  get_or_create <- function(n) {
    cache_path <- if (is.null(cache_dir)) NULL else file.path(cache_dir, sprintf("inputs_%s.rds", n))
    if (!is.null(cache_path) && file.exists(cache_path)) {
      return(load_from_cache(cache_path, n))
    }
    regenerate(n)
  }

  lapply(sizes, get_or_create)
}

default_min_time <- 0.25
default_min_iterations <- 3L

distribution_operations <- function() {
  list(
    list(
      id = "dnorm",
      label = "dnorm",
      base = function(data) { dnorm(data$vec); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_dnorm(data$vec)) }
    ),
    list(
      id = "pnorm",
      label = "pnorm",
      base = function(data) { pnorm(data$vec); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_pnorm(data$vec)) }
    ),
    list(
      id = "qnorm",
      label = "qnorm",
      base = function(data) { qnorm(data$prob); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_qnorm(data$prob)) }
    ),
    list(
      id = "dunif",
      label = "dunif",
      base = function(data) { dunif(data$vec); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_dunif(data$vec)) }
    ),
    list(
      id = "punif",
      label = "punif",
      base = function(data) { punif(data$vec); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_punif(data$vec)) }
    ),
    list(
      id = "qunif",
      label = "qunif",
      base = function(data) { qunif(data$prob); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_qunif(data$prob)) }
    ),
    list(
      id = "dexp",
      label = "dexp",
      base = function(data) { dexp(abs(data$vec)); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_dexp(abs(data$vec))) }
    ),
    list(
      id = "pexp",
      label = "pexp",
      base = function(data) { pexp(abs(data$vec)); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_pexp(abs(data$vec))) }
    ),
    list(
      id = "qexp",
      label = "qexp",
      base = function(data) { qexp(data$prob); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_qexp(data$prob)) }
    ),
    list(
      id = "dlnorm",
      label = "dlnorm",
      base = function(data) { dlnorm(abs(data$vec)); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_dlnorm(abs(data$vec))) }
    ),
    list(
      id = "plnorm",
      label = "plnorm",
      base = function(data) { plnorm(abs(data$vec)); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_plnorm(abs(data$vec))) }
    ),
    list(
      id = "qlnorm",
      label = "qlnorm",
      base = function(data) { qlnorm(data$prob); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_qlnorm(data$prob)) }
    ),
    list(
      id = "dlogis",
      label = "dlogis",
      base = function(data) { dlogis(data$vec); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_dlogis(data$vec)) }
    ),
    list(
      id = "plogis",
      label = "plogis",
      base = function(data) { plogis(data$vec); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_plogis(data$vec)) }
    ),
    list(
      id = "qlogis",
      label = "qlogis",
      base = function(data) { qlogis(data$prob); invisible(NULL) },
      mlx = function(data) { force_mlx(mlx_qlogis(data$prob)) }
    )
  )
}

benchmark_operations <- function() {
  list(
    list(
      id = "matmul",
      label = "Matrix multiply",
      base = function(data) { data$a %*% data$b; invisible(NULL) },
      mlx = function(data) { force_mlx(data$a %*% data$b) }
    ),
    list(
      id = "add",
      label = "Matrix addition",
      base = function(data) { data$a + data$b; invisible(NULL) },
      mlx = function(data) { force_mlx(data$a + data$b) }
    ),
    list(
      id = "subset_vec",
      label = "Subset rows (vector)",
      base = function(data) { data$a[data$idx_vec, , drop = FALSE]; invisible(NULL) },
      mlx = function(data) { force_mlx(data$a[data$idx_vec, , drop = FALSE]) }
    ),
    list(
      id = "subset_mat",
      label = "Subset (matrix index)",
      base = function(data) { data$a[data$idx_mat]; invisible(NULL) },
      mlx = function(data) { force_mlx(data$a[data$idx_mat]) }
    ),
    list(
      id = "sum",
      label = "Sum",
      base = function(data) { sum(data$a); invisible(NULL) },
      mlx = function(data) { force_mlx(sum(data$a)) }
    ),
    list(
      id = "mean",
      label = "Mean",
      base = function(data) { mean(data$a); invisible(NULL) },
      mlx = function(data) { force_mlx(mean(data$a)) }
    ),
    list(
      id = "rowsums",
      label = "Row sums",
      base = function(data) { rowSums(data$a); invisible(NULL) },
      mlx = function(data) { force_mlx(rowSums(data$a)) }
    ),
    list(
      id = "rowmeans",
      label = "Row means",
      base = function(data) { rowMeans(data$a); invisible(NULL) },
      mlx = function(data) { force_mlx(rowMeans(data$a)) }
    ),
    list(
      id = "tcrossprod",
      label = "tcrossprod",
      base = function(data) { tcrossprod(data$a); invisible(NULL) },
      mlx = function(data) { force_mlx(tcrossprod(data$a)) }
    ),
    list(
      id = "scale",
      label = "scale()",
      base = function(data) { scale(data$a); invisible(NULL) },
      mlx = function(data) { force_mlx(scale(data$a)) }
    ),
    list(
      id = "solve",
      label = "Solve Ax = b",
      base = function(data) { solve(data$spd, data$rhs); invisible(NULL) },
      mlx = function(data) { force_mlx(solve(data$spd, data$rhs)) }
    ),
    list(
      id = "backsolve",
      label = "Backsolve",
      base = function(data) { backsolve(data$chol, data$rhs); invisible(NULL) },
      mlx = function(data) { force_mlx(backsolve(data$chol, data$rhs)) }
    ),
    list(
      id = "chol",
      label = "Cholesky",
      base = function(data) { chol(data$spd); invisible(NULL) },
      mlx = function(data) { force_mlx(chol(data$spd)) }
    ),
    list(
      id = "chol2inv",
      label = "chol2inv",
      base = function(data) { chol2inv(data$chol); invisible(NULL) },
      mlx = function(data) { force_mlx(chol2inv(data$chol)) }
    ),
    list(
      id = "svd",
      label = "SVD (values only)",
      min_iterations = 1L,
      base = function(data) { svd(data$a, nu = 0, nv = 0); invisible(NULL) },
      mlx = function(data) { force_mlx(svd(data$a, nu = 0, nv = 0)) }
    ),
    list(
      id = "diag",
      label = "Diagonal",
      base = function(data) { diag(data$a); invisible(NULL) },
      mlx = function(data) { force_mlx(diag(data$a)) }
    )
  )
}

run_benchmarks <- function(operations, inputs) {
  rows <- vector("list", length(operations) * length(inputs))
  idx <- 1L
  for (op in operations) {
    for (size_name in names(inputs)) {
      bench_res <- bench::mark(
        base = op$base(inputs[[size_name]]$base),
        mlx = op$mlx(inputs[[size_name]]$mlx),
        min_time = op$min_time %||% default_min_time,
        min_iterations = op$min_iterations %||% default_min_iterations,
        check = FALSE,
        filter_gc = FALSE
      )
      rows[[idx]] <- data.frame(
        operation = op$label,
        size = size_name,
        implementation = as.character(bench_res$expression),
        median_seconds = as.numeric(bench_res$median, units = "secs"),
        itr_per_sec = bench_res$`itr/sec`,
        mem_alloc_bytes = as.numeric(bench_res$mem_alloc, units = "bytes"),
        stringsAsFactors = FALSE
      )
      idx <- idx + 1L
    }
  }
  do.call(rbind, rows)
}
