#!/usr/bin/env Rscript

# Script: dev/matrix_function_compat.R
# Purpose: Systematically test how base R matrix-oriented functions behave with mlx objects.
# Output: A CSV written to docs/matrix_function_compat_results.csv (unless overridden via --output).

args <- commandArgs(trailingOnly = TRUE)
output_path <- "docs/matrix_function_compat_results.csv"
as_matrix_mode <- "default"
parse_arg <- function(arg) {
  if (grepl("^--output=", arg)) {
    return(list(type = "output", value = sub("^--output=", "", arg)))
  }
  if (grepl("^--mode=", arg)) {
    return(list(type = "mode", value = sub("^--mode=", "", arg)))
  }
  list(type = "output", value = arg)
}
if (length(args) > 0) {
  parsed <- lapply(args, parse_arg)
  for (entry in parsed) {
    if (entry$type == "output") {
      output_path <- entry$value
    } else if (entry$type == "mode") {
      as_matrix_mode <- entry$value
    }
  }
}

suppressPackageStartupMessages({
  if (!requireNamespace("pkgload", quietly = TRUE)) {
    stop("pkgload is required to run this script. Install it with install.packages('pkgload').")
  }
  pkgload::load_all(".", export_all = FALSE, helpers = FALSE, attach_testthat = FALSE)
})

if (identical(as_matrix_mode, "identity")) {
  orig_fun <- getS3method("as.matrix", "mlx")
  identity_fun <- function(x, ...) x
  environment(identity_fun) <- asNamespace("Rmlx")
  base_ns <- asNamespace("base")
  registerS3method("as.matrix", "mlx", identity_fun, envir = base_ns)
  rmlx_ns <- asNamespace("Rmlx")
  if (bindingIsLocked("as.matrix.mlx", rmlx_ns)) {
    unlockBinding("as.matrix.mlx", rmlx_ns)
    on.exit(lockBinding("as.matrix.mlx", rmlx_ns), add = TRUE)
  }
  assign("as.matrix.mlx", identity_fun, envir = rmlx_ns)
  assign("as.matrix.mlx", identity_fun, envir = globalenv())
  on.exit({
    registerS3method("as.matrix", "mlx", orig_fun, envir = base_ns)
    assign("as.matrix.mlx", orig_fun, envir = rmlx_ns)
    assign("as.matrix.mlx", orig_fun, envir = globalenv())
  }, add = TRUE)
  message("Temporarily overriding as.matrix.mlx() to identity for compatibility run.")
}

## Fixtures --------------------------------------------------------------------

make_matrix <- function(vals, nrow, ncol) {
  matrix(vals, nrow = nrow, ncol = ncol, byrow = FALSE)
}

base_mat <- make_matrix(1:12, nrow = 3, ncol = 4)
base_mat2 <- make_matrix(seq(2, 24, by = 2), nrow = 3, ncol = 4)
base_mat_negpos <- make_matrix(seq(-6, 5, length.out = 12), nrow = 3, ncol = 4)
unit_mat <- make_matrix(seq(-0.9, 0.9, length.out = 12), nrow = 3, ncol = 4)
bounded_mat <- make_matrix(seq(-0.8, 0.8, length.out = 12), nrow = 3, ncol = 4)
positive_mat <- make_matrix(seq(0.25, 3.25, length.out = 12), nrow = 3, ncol = 4)
prob_mat <- make_matrix(seq(0.05, 0.95, length.out = 12), nrow = 3, ncol = 4)
square_pd <- matrix(c(4, 1, 1, 3), nrow = 2)
upper_tri <- matrix(c(3, 1, 0, 2), nrow = 2)
lower_tri <- t(upper_tri)
rhs_vec <- c(1, 2)
rhs_mat <- matrix(c(1, 0, 0, 1), nrow = 2)

mx <- as_mlx(base_mat)
mx2 <- as_mlx(base_mat2)
mx_negpos <- as_mlx(base_mat_negpos)
mx_unit <- as_mlx(unit_mat)
mx_bounded <- as_mlx(bounded_mat)
mx_positive <- as_mlx(positive_mat)
mx_prob <- as_mlx(prob_mat)
mx_square <- as_mlx(square_pd)
mx_upper <- as_mlx(upper_tri)
mx_lower <- as_mlx(lower_tri)
mx_rhs_vec <- as_mlx(rhs_vec)
mx_rhs_mat <- as_mlx(rhs_mat)

dimnames_list <- list(
  rows = paste0("r", seq_len(nrow(base_mat))),
  cols = paste0("c", seq_len(ncol(base_mat)))
)

group_rows <- factor(c("A", "A", "B"))
stats_rows <- c(1, 2, 3)
stats_cols <- c(10, 20, 30, 40)

logical_mx <- mx > 5
logical_mx2 <- mx2 > 8

list_for_simplify <- list(mx, mx2)

qr_rhs <- c(1, 2)

data_env <- list2env(list(
  mx = mx,
  mx2 = mx2,
  mx_negpos = mx_negpos,
  mx_unit = mx_unit,
  mx_bounded = mx_bounded,
  mx_positive = mx_positive,
  mx_prob = mx_prob,
  mx_square = mx_square,
  mx_upper = mx_upper,
  mx_lower = mx_lower,
  mx_rhs_vec = mx_rhs_vec,
  mx_rhs_mat = mx_rhs_mat,
  base_mat = base_mat,
  base_mat2 = base_mat2,
  square_pd = square_pd,
  upper_tri = upper_tri,
  lower_tri = lower_tri,
  rhs_vec = rhs_vec,
  rhs_mat = rhs_mat,
  dimnames_list = dimnames_list,
  group_rows = group_rows,
  stats_rows = stats_rows,
  stats_cols = stats_cols,
  logical_mx = logical_mx,
  logical_mx2 = logical_mx2,
  list_for_simplify = list_for_simplify,
  qr_rhs = qr_rhs
), parent = globalenv())

## Helpers ---------------------------------------------------------------------

cases <- list()

add_case <- function(name, expr, category, note = NA_character_) {
  cases[[length(cases) + 1L]] <<- list(
    name = name,
    category = category,
    expr = expr,
    note = note
  )
}

add_unary_cases <- function(fn_names, input_sym, category) {
  for (fn in fn_names) {
    expr <- call(fn, as.name(input_sym))
    add_case(fn, expr, category)
  }
}

add_binary_cases <- function(fn_names, lhs_sym, rhs_sym, category) {
  for (fn in fn_names) {
    expr <- call(fn, as.name(lhs_sym), as.name(rhs_sym))
    add_case(fn, expr, category)
  }
}

add_operator_cases <- function(operators, lhs_sym, rhs_sym, category) {
  for (op in operators) {
    expr <- call(op, as.name(lhs_sym), as.name(rhs_sym))
    add_case(op, expr, category)
  }
}

## Case definitions ------------------------------------------------------------

# 1. Construction, reshaping, metadata
add_case("matrix", quote(matrix(mx, nrow = nrow(base_mat))), "construction")
add_case("array", quote(array(mx, dim = dim(base_mat))), "construction")
add_case("as.matrix", quote(as.matrix(mx)), "construction")
add_case("as.array", quote(as.array(mx)), "construction")
add_case("dim", quote(dim(mx)), "construction")
add_case("dim<-", quote({ tmp <- mx; dim(tmp) <- rev(dim(base_mat)); tmp }), "construction")
add_case("dimnames", quote(dimnames(mx)), "construction")
add_case("dimnames<-", quote({ tmp <- mx; dimnames(tmp) <- dimnames_list; tmp }), "construction")
add_case("rownames", quote(rownames(mx)), "construction")
add_case("rownames<-", quote({ tmp <- mx; rownames(tmp) <- dimnames_list$rows; tmp }), "construction")
add_case("colnames", quote(colnames(mx)), "construction")
add_case("colnames<-", quote({ tmp <- mx; colnames(tmp) <- dimnames_list$cols; tmp }), "construction")
add_case("drop", quote(drop(mx)), "construction")
add_case("t", quote(t(mx)), "construction")
add_case("aperm", quote(aperm(mx, perm = c(2, 1))), "construction")
add_case("cbind", quote(cbind(mx, mx2)), "construction")
add_case("rbind", quote(rbind(mx, mx2)), "construction")
add_case("diag", quote(diag(mx_square)), "construction")
add_case("diag<-", quote({ tmp <- mx_square; diag(tmp) <- 0; tmp }), "construction")
add_case("lower.tri", quote(lower.tri(mx_square)), "construction")
add_case("upper.tri", quote(upper.tri(mx_square)), "construction")
add_case("row", quote(row(mx)), "construction")
add_case("col", quote(col(mx)), "construction")
add_case("replicate", quote(replicate(2, mx)), "construction")
add_case("simplify2array", quote(simplify2array(list_for_simplify)), "construction")

# 2. Row/column aggregations
add_case("apply_rows", quote(apply(mx, 1, mean)), "rowcol", note = "apply margin=1")
add_case("apply_cols", quote(apply(mx, 2, mean)), "rowcol", note = "apply margin=2")
add_case("rowSums", quote(rowSums(mx)), "rowcol")
add_case("colSums", quote(colSums(mx)), "rowcol")
add_case("rowMeans", quote(rowMeans(mx)), "rowcol")
add_case("colMeans", quote(colMeans(mx)), "rowcol")
add_case("rowsum", quote(rowsum(mx, group_rows)), "rowcol")
add_case("sweep_rows", quote(sweep(mx, 1, stats_rows, FUN = "+")), "rowcol")
add_case("sweep_cols", quote(sweep(mx, 2, stats_cols, FUN = "-")), "rowcol")
add_case("scale", quote(scale(mx)), "rowcol")
add_case("prop.table_rows", quote(prop.table(mx, margin = 1)), "rowcol")
add_case("prop.table_cols", quote(prop.table(mx, margin = 2)), "rowcol")
add_case("margin.table_rows", quote(margin.table(mx, margin = 1)), "rowcol")
add_case("margin.table_cols", quote(margin.table(mx, margin = 2)), "rowcol")
add_case("addmargins_rows", quote(addmargins(mx, margin = 1)), "rowcol")
add_case("addmargins_cols", quote(addmargins(mx, margin = 2)), "rowcol")
add_case("max.col", quote(max.col(mx)), "rowcol")
add_case("duplicated.matrix", quote(duplicated.matrix(mx)), "rowcol")
add_case("unique.matrix", quote(unique.matrix(mx)), "rowcol")
add_case("anyDuplicated.matrix", quote(anyDuplicated.matrix(mx)), "rowcol")

# 3. Elementwise Ops
arithmetic_ops <- c("+", "-", "*", "/", "^", "%/%", "%%")
add_operator_cases(arithmetic_ops, "mx", "mx2", "elementwise_ops")

comparison_ops <- c("<", "<=", ">", ">=", "==", "!=")
add_operator_cases(comparison_ops, "mx", "mx2", "elementwise_ops")

logical_ops <- c("&", "|")
add_operator_cases(logical_ops, "logical_mx", "logical_mx2", "elementwise_ops")

add_case("xor", quote(xor(logical_mx, logical_mx2)), "elementwise_ops")
add_case("!", quote(!logical_mx), "elementwise_ops")
add_case("ifelse", quote(ifelse(logical_mx, mx, mx2)), "elementwise_ops")
add_case("pmin", quote(pmin(mx, mx2)), "elementwise_ops")
add_case("pmax", quote(pmax(mx, mx2)), "elementwise_ops")
add_case("pmin.int", quote(pmin.int(mx, mx2)), "elementwise_ops")
add_case("pmax.int", quote(pmax.int(mx, mx2)), "elementwise_ops")

# 4. Elementwise Math
math_inputs <- list(
  default = "mx_negpos",
  positive = "mx_positive",
  bounded = "mx_bounded",
  unit = "mx_unit",
  prob = "mx_prob"
)

math_general <- c("abs", "sign", "floor", "ceiling", "trunc",
                  "round", "signif", "sin", "cos", "tan",
                  "sinh", "cosh", "tanh")
add_unary_cases(math_general, math_inputs$default, "elementwise_math")

math_positive <- c("sqrt", "exp", "expm1", "log", "log10",
                   "log2", "log1p", "gamma", "lgamma", "digamma", "trigamma")
add_unary_cases(math_positive, math_inputs$positive, "elementwise_math")

math_unit_domain <- c("asin", "acos", "atan")
add_unary_cases(math_unit_domain, math_inputs$unit, "elementwise_math")

add_case("atan2", quote(atan2(mx_negpos, mx_positive)), "elementwise_math")
add_case("sinpi", quote(sinpi(mx)), "elementwise_math")
add_case("cospi", quote(cospi(mx)), "elementwise_math")
add_case("tanpi", quote(tanpi(mx)), "elementwise_math")
add_case("asinh", quote(asinh(mx_negpos)), "elementwise_math")
add_case("acosh", quote(acosh(mx_positive + 1)), "elementwise_math")
add_case("atanh", quote(atanh(mx_bounded / 2)), "elementwise_math")

# 5. Complex helpers
add_unary_cases(c("Re", "Im", "Mod", "Arg", "Conj"), math_inputs$positive, "complex_helpers")

# 6. Elementwise predicates
add_unary_cases(c("is.na", "is.nan", "is.finite", "is.infinite"), "mx", "predicates")

# 7. Matrix algebra and decompositions
add_case("%*%", quote(mx %*% t(mx2)), "linalg")
add_case("crossprod", quote(crossprod(mx, mx2)), "linalg")
add_case("tcrossprod", quote(tcrossprod(mx, mx2)), "linalg")
add_case("solve_mat", quote(solve(mx_square)), "linalg")
add_case("solve_rhs_vec", quote(solve(mx_square, rhs_vec)), "linalg")
add_case("solve_rhs_mx", quote(solve(mx_square, mx_rhs_mat)), "linalg")
add_case("backsolve", quote(backsolve(mx_upper, rhs_vec)), "linalg")
add_case("forwardsolve", quote(forwardsolve(mx_lower, rhs_vec)), "linalg")
add_case("chol", quote(chol(mx_square)), "linalg")
add_case("chol2inv", quote(chol2inv(chol(mx_square))), "linalg")
add_case("det", quote(det(mx_square)), "linalg")
add_case("determinant", quote(determinant(mx_square)), "linalg")
add_case("qr", quote(qr(mx_square)), "linalg")
add_case("qr.Q", quote(qr.Q(qr(mx_square))), "linalg")
add_case("qr.R", quote(qr.R(qr(mx_square))), "linalg")
add_case("qr.fitted", quote(qr.fitted(qr(mx_square), qr_rhs)), "linalg")
add_case("qr.resid", quote(qr.resid(qr(mx_square), qr_rhs)), "linalg")
add_case("qr.solve", quote(qr.solve(mx_square, qr_rhs)), "linalg")
add_case("qr.qty", quote(qr.qty(qr(mx_square), qr_rhs)), "linalg")
add_case("qr.qy", quote(qr.qy(qr(mx_square), qr_rhs)), "linalg")
add_case("eigen", quote(eigen(mx_square)), "linalg")
add_case("svd", quote(svd(mx)), "linalg")
add_case("norm", quote(norm(mx_square)), "linalg")
add_case("kappa", quote(kappa(mx_square)), "linalg")
add_case("rcond", quote(rcond(mx_square)), "linalg")
add_case("fft", quote(fft(mx)), "linalg")
add_case("mvfft", quote(mvfft(mx)), "linalg")

# 8. Structured combinations
add_case("outer", quote(outer(mx, mx2)), "structured")
add_case("kronecker", quote(kronecker(mx_square, mx_square)), "structured")
add_case("%o%", quote(mx %o% mx2), "structured")

# 9. Indexing helpers
add_case("which_arr_ind", quote(which(mx > 5, arr.ind = TRUE)), "indexing")
add_case("arrayInd", quote(arrayInd(which(mx > 5), dim(mx))), "indexing")

## Evaluation ------------------------------------------------------------------

evaluate_case <- function(case, env) {
  call_txt <- paste(deparse(case$expr, width.cutoff = 120), collapse = " ")
  status <- "ok"
  note <- case$note
  value <- tryCatch(
    eval(case$expr, env),
    error = function(e) {
      status <<- "error"
      note <<- trimws(paste(note, conditionMessage(e)))
      NULL
    }
  )
  returns_mlx <- if (!is.null(value)) inherits(value, "mlx") else NA
  result_class <- if (!is.null(value)) paste(class(value), collapse = ":") else NA
  result_dim <- if (!is.null(value) && !is.null(dim(value))) paste(dim(value), collapse = "x") else NA
  list(
    function_name = case$name,
    category = case$category,
    call = call_txt,
    status = status,
    returns_mlx = returns_mlx,
    result_class = result_class,
    result_dim = result_dim,
    note = if (is.na(note)) "" else note
  )
}

results_list <- lapply(cases, evaluate_case, env = data_env)
results_df <- do.call(
  rbind,
  lapply(results_list, function(x) {
    data.frame(
      function_name = x$function_name,
      category = x$category,
      call = x$call,
      status = x$status,
      returns_mlx = x$returns_mlx,
      result_class = x$result_class,
      result_dim = x$result_dim,
      note = x$note,
      as_matrix_mode = as_matrix_mode,
      stringsAsFactors = FALSE
    )
  })
)

dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
write.csv(results_df, file = output_path, row.names = FALSE)

message("Wrote ", nrow(results_df), " compatibility rows to ", output_path)
