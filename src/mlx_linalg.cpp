// Linear algebra operations
#include "mlx_helpers.hpp"
#include <mlx/mlx.h>
#include <mlx/linalg.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

// [[Rcpp::export]]
SEXP cpp_mlx_solve(SEXP a_xp_, SEXP b_xp_,
                   std::string dtype_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  array a_target = astype(a_wrapper->get(), target_dtype);

  array result = [&]() -> array {
    if (b_xp_ == R_NilValue) {
      // No b provided: compute matrix inverse
      return linalg::inv(a_target);
    } else {
      // b provided: solve linear system Ax = b
      MlxArrayWrapper* b_wrapper = get_mlx_wrapper(b_xp_);
      array b_target = astype(b_wrapper->get(), target_dtype);
      return linalg::solve(a_target, b_target);
    }
  }();

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_cholesky(SEXP a_xp_, bool upper,
                      std::string dtype_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);

  array a_target = astype(a_wrapper->get(), target_dtype);
  array chol_target = mlx::core::linalg::cholesky(a_target, upper);

  return make_mlx_xptr(std::move(chol_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_qr(SEXP a_xp_,
                std::string dtype_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);

  array a_target = astype(a_wrapper->get(), target_dtype);
  auto qr_target = mlx::core::linalg::qr(a_target);

  array q_target = astype(qr_target.first, target_dtype);
  array r_target = astype(qr_target.second, target_dtype);

  return List::create(
      Named("Q") = make_mlx_xptr(std::move(q_target)),
      Named("R") = make_mlx_xptr(std::move(r_target)));
}

// [[Rcpp::export]]
SEXP cpp_mlx_svd(SEXP a_xp_, bool compute_uv,
                 std::string dtype_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);

  array a_target = astype(a_wrapper->get(), target_dtype);
  std::vector<array> svd_target = mlx::core::linalg::svd(a_target, compute_uv, {});

  List out(svd_target.size());
  CharacterVector names(svd_target.size());
  if (svd_target.size() == 3) {
    names[0] = "U";
    names[1] = "S";
    names[2] = "Vh";
  } else if (svd_target.size() == 2) {
    names[0] = "S";
    names[1] = "Vh";
  } else if (svd_target.size() == 1) {
    names[0] = "S";
  }

  for (size_t i = 0; i < svd_target.size(); ++i) {
    array target = astype(svd_target[i], target_dtype);
    out[i] = make_mlx_xptr(std::move(target));
  }

  if (svd_target.size() > 0) {
    out.attr("names") = names;
  }

  return out;
}

// [[Rcpp::export]]
SEXP cpp_mlx_pinv(SEXP a_xp_,
                  std::string dtype_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);

  array a_target = astype(a_wrapper->get(), target_dtype);
  array pinv_target = mlx::core::linalg::pinv(a_target);

  return make_mlx_xptr(std::move(pinv_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_norm(SEXP xp_, SEXP ord_,
                  Rcpp::Nullable<Rcpp::IntegerVector> axes,
                  bool keepdims) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();


  array arr_target = astype(arr, arr.dtype());
  std::optional<std::vector<int>> axes_opt = optional_axes(arr, axes);

  array result = [&]() -> array {
    if (Rf_isNull(ord_)) {
      if (axes_opt.has_value()) {
        return mlx::core::linalg::norm(arr_target, axes_opt.value(), keepdims);
      }
      return mlx::core::linalg::norm(arr_target, std::nullopt, keepdims);
    }
    if (Rf_isReal(ord_) || Rf_isInteger(ord_)) {
      double ord_val = Rcpp::as<double>(ord_);
      if (axes_opt.has_value()) {
        return mlx::core::linalg::norm(arr_target, ord_val, axes_opt.value(), keepdims);
      }
      return mlx::core::linalg::norm(arr_target, ord_val, std::nullopt, keepdims);
    }
    if (Rf_isString(ord_)) {
      std::string ord_str = Rcpp::as<std::string>(ord_);
      if (axes_opt.has_value()) {
        return mlx::core::linalg::norm(arr_target, ord_str, axes_opt.value(), keepdims);
      }
      return mlx::core::linalg::norm(arr_target, ord_str, std::nullopt, keepdims);
    }
    Rcpp::stop("Unsupported ord type for mlx_norm.");
  }();

  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_eig(SEXP xp_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();


  array arr_target = astype(arr, arr.dtype());
  auto eig_pair = mlx::core::linalg::eig(arr_target);

  array values_target = astype(eig_pair.first, eig_pair.first.dtype());
  array vectors_target = astype(eig_pair.second, eig_pair.second.dtype());

  return List::create(
      Named("values") = make_mlx_xptr(std::move(values_target)),
      Named("vectors") = make_mlx_xptr(std::move(vectors_target)));
}

// [[Rcpp::export]]
SEXP cpp_mlx_eigvals(SEXP xp_) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();


  array arr_target = astype(arr, arr.dtype());
  array vals_target = mlx::core::linalg::eigvals(arr_target);
  return make_mlx_xptr(std::move(vals_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_eigvalsh(SEXP xp_, std::string uplo) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();


  array arr_target = astype(arr, arr.dtype());
  array vals_target = mlx::core::linalg::eigvalsh(arr_target, uplo);
  return make_mlx_xptr(std::move(vals_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_eigh(SEXP xp_, std::string uplo) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();


  array arr_target = astype(arr, arr.dtype());
  auto eig_pair = mlx::core::linalg::eigh(arr_target, uplo);

  array values_target = astype(eig_pair.first, eig_pair.first.dtype());
  array vectors_target = astype(eig_pair.second, eig_pair.second.dtype());

  return List::create(
      Named("values") = make_mlx_xptr(std::move(values_target)),
      Named("vectors") = make_mlx_xptr(std::move(vectors_target)));
}

// [[Rcpp::export]]
SEXP cpp_mlx_solve_triangular(SEXP a_xp_, SEXP b_xp_, bool upper) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);
  MlxArrayWrapper* b_wrapper = get_mlx_wrapper(b_xp_);

  array a_arr = a_wrapper->get();
  array b_arr = b_wrapper->get();
  Dtype target_dtype = promote_numeric_dtype(a_arr.dtype(), b_arr.dtype());

  array a_target = astype(a_arr, target_dtype);
  array b_target = astype(b_arr, target_dtype);

  array result = mlx::core::linalg::solve_triangular(
    a_target, b_target, upper);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_cross(SEXP a_xp_, SEXP b_xp_, int axis) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);
  MlxArrayWrapper* b_wrapper = get_mlx_wrapper(b_xp_);

  array a_arr = a_wrapper->get();
  array b_arr = b_wrapper->get();
  Dtype target_dtype = promote_numeric_dtype(a_arr.dtype(), b_arr.dtype());

  array a_target = astype(a_arr, target_dtype);
  array b_target = astype(b_arr, target_dtype);

  int axis_input = axis;
  if (axis_input >= 0) {
    axis_input -= 1;
  }
  int ax = normalize_axis(a_target, axis_input);
  array result = mlx::core::linalg::cross(a_target, b_target, ax);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_trace(SEXP a_xp_, int offset, int axis1, int axis2) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  array arr = a_wrapper->get();

  // Convert 1-indexed to 0-indexed
  int ax1 = axis1 - 1;
  int ax2 = axis2 - 1;

  ax1 = normalize_axis(arr, ax1);
  ax2 = normalize_axis(arr, ax2);

  array result = trace(arr, offset, ax1, ax2);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_diagonal(SEXP a_xp_, int offset, int axis1, int axis2) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  array arr = a_wrapper->get();

  // Convert 1-indexed to 0-indexed
  int ax1 = axis1 - 1;
  int ax2 = axis2 - 1;

  ax1 = normalize_axis(arr, ax1);
  ax2 = normalize_axis(arr, ax2);

  array result = diagonal(arr, offset, ax1, ax2);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_diag(SEXP a_xp_, int k) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  array arr = a_wrapper->get();

  array result = diag(arr, k);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_outer(SEXP a_xp_, SEXP b_xp_) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);
  MlxArrayWrapper* b_wrapper = get_mlx_wrapper(b_xp_);

  array a_arr = a_wrapper->get();
  array b_arr = b_wrapper->get();
  Dtype target_dtype = promote_numeric_dtype(a_arr.dtype(), b_arr.dtype());

  array a_target = astype(a_arr, target_dtype);
  array b_target = astype(b_arr, target_dtype);

  array result = outer(a_target, b_target);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP cpp_mlx_inv(SEXP a_xp_) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  array arr = a_wrapper->get();

  array a_target = astype(arr, arr.dtype());
  array result_target = linalg::inv(a_target);

  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_tri_inv(SEXP a_xp_, bool upper) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  array arr = a_wrapper->get();

  array a_target = astype(arr, arr.dtype());
  array result_target = linalg::tri_inv(a_target, upper);

  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_cholesky_inv(SEXP a_xp_, bool upper) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  array arr = a_wrapper->get();

  array a_target = astype(arr, arr.dtype());
  array result_target = linalg::cholesky_inv(a_target, upper);

  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_lu(SEXP a_xp_) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  array arr = a_wrapper->get();

  array a_target = astype(arr, arr.dtype());

  // lu() returns a vector of arrays [P, L, U]
  auto lu_result = linalg::lu(a_target);

  if (lu_result.size() != 3) {
    Rcpp::stop("Unexpected LU result size");
  }

  array p_target = lu_result[0];  // Pivot indices
  array l_target = lu_result[1];  // Lower triangular
  array u_target = lu_result[2];  // Upper triangular

  return List::create(
    Named("p") = make_mlx_xptr(std::move(p_target)),
    Named("l") = make_mlx_xptr(std::move(l_target)),
    Named("u") = make_mlx_xptr(std::move(u_target))
  );
}

// [[Rcpp::export]]
SEXP cpp_mlx_kron(SEXP a_xp_, SEXP b_xp_) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);
  MlxArrayWrapper* b_wrapper = get_mlx_wrapper(b_xp_);

  array a_arr = a_wrapper->get();
  array b_arr = b_wrapper->get();

  array result = kron(a_arr, b_arr);
  return make_mlx_xptr(std::move(result));
}
