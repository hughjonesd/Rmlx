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
                   std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);
  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);

  array result = [&]() -> array {
    if (b_xp_ == R_NilValue) {
      // No b provided: compute matrix inverse
      return linalg::inv(a_cpu, cpu_stream);
    } else {
      // b provided: solve linear system Ax = b
      MlxArrayWrapper* b_wrapper = get_mlx_wrapper(b_xp_);
      array b_cpu = astype(b_wrapper->get(), target_dtype, cpu_stream);
      return linalg::solve(a_cpu, b_cpu, cpu_stream);
    }
  }();

array result_target = astype(result, target_dtype, target_device);

  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_cholesky(SEXP a_xp_, bool upper,
                      std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);

  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);
  array chol_cpu = mlx::core::linalg::cholesky(a_cpu, upper, cpu_stream);
  array chol_target = astype(chol_cpu, target_dtype, target_device);

  return make_mlx_xptr(std::move(chol_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_qr(SEXP a_xp_,
                std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);

  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);
  auto qr_cpu = mlx::core::linalg::qr(a_cpu, cpu_stream);

  array q_target = astype(qr_cpu.first, target_dtype, target_device);
  array r_target = astype(qr_cpu.second, target_dtype, target_device);

  return List::create(
      Named("Q") = make_mlx_xptr(std::move(q_target)),
      Named("R") = make_mlx_xptr(std::move(r_target)));
}

// [[Rcpp::export]]
SEXP cpp_mlx_svd(SEXP a_xp_, bool compute_uv,
                 std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);

  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);
  std::vector<array> svd_cpu = mlx::core::linalg::svd(a_cpu, compute_uv, cpu_stream);

  List out(svd_cpu.size());
  CharacterVector names(svd_cpu.size());
  if (svd_cpu.size() == 3) {
    names[0] = "U";
    names[1] = "S";
    names[2] = "Vh";
  } else if (svd_cpu.size() == 2) {
    names[0] = "S";
    names[1] = "Vh";
  } else if (svd_cpu.size() == 1) {
    names[0] = "S";
  }

  for (size_t i = 0; i < svd_cpu.size(); ++i) {
    array target = astype(svd_cpu[i], target_dtype, target_device);
    out[i] = make_mlx_xptr(std::move(target));
  }

  if (svd_cpu.size() > 0) {
    out.attr("names") = names;
  }

  return out;
}

// [[Rcpp::export]]
SEXP cpp_mlx_pinv(SEXP a_xp_,
                  std::string dtype_str, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  Dtype target_dtype = string_to_dtype(dtype_str);
  StreamOrDevice target_device = string_to_device(device_str);
  StreamOrDevice cpu_stream = Device(Device::cpu);

  array a_cpu = astype(a_wrapper->get(), target_dtype, cpu_stream);
  array pinv_cpu = mlx::core::linalg::pinv(a_cpu, cpu_stream);
  array pinv_target = astype(pinv_cpu, target_dtype, target_device);

  return make_mlx_xptr(std::move(pinv_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_norm(SEXP xp_, SEXP ord_,
                  Rcpp::Nullable<Rcpp::IntegerVector> axes,
                  bool keepdims, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array arr_cpu = astype(arr, arr.dtype(), cpu_stream);
  std::optional<std::vector<int>> axes_opt = optional_axes(arr, axes);

  array result_cpu = [&]() -> array {
    if (Rf_isNull(ord_)) {
      if (axes_opt.has_value()) {
        return mlx::core::linalg::norm(arr_cpu, axes_opt.value(), keepdims, cpu_stream);
      }
      return mlx::core::linalg::norm(arr_cpu, std::nullopt, keepdims, cpu_stream);
    }
    if (Rf_isReal(ord_) || Rf_isInteger(ord_)) {
      double ord_val = Rcpp::as<double>(ord_);
      if (axes_opt.has_value()) {
        return mlx::core::linalg::norm(arr_cpu, ord_val, axes_opt.value(), keepdims, cpu_stream);
      }
      return mlx::core::linalg::norm(arr_cpu, ord_val, std::nullopt, keepdims, cpu_stream);
    }
    if (Rf_isString(ord_)) {
      std::string ord_str = Rcpp::as<std::string>(ord_);
      if (axes_opt.has_value()) {
        return mlx::core::linalg::norm(arr_cpu, ord_str, axes_opt.value(), keepdims, cpu_stream);
      }
      return mlx::core::linalg::norm(arr_cpu, ord_str, std::nullopt, keepdims, cpu_stream);
    }
    Rcpp::stop("Unsupported ord type for mlx_norm.");
  }();

  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_eig(SEXP xp_, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array arr_cpu = astype(arr, arr.dtype(), cpu_stream);
  auto eig_pair = mlx::core::linalg::eig(arr_cpu, cpu_stream);

  array values_target = astype(eig_pair.first, eig_pair.first.dtype(), target_device);
  array vectors_target = astype(eig_pair.second, eig_pair.second.dtype(), target_device);

  return List::create(
      Named("values") = make_mlx_xptr(std::move(values_target)),
      Named("vectors") = make_mlx_xptr(std::move(vectors_target)));
}

// [[Rcpp::export]]
SEXP cpp_mlx_eigvals(SEXP xp_, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array arr_cpu = astype(arr, arr.dtype(), cpu_stream);
  array vals_cpu = mlx::core::linalg::eigvals(arr_cpu, cpu_stream);
  array vals_target = astype(vals_cpu, vals_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(vals_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_eigvalsh(SEXP xp_, std::string uplo, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array arr_cpu = astype(arr, arr.dtype(), cpu_stream);
  array vals_cpu = mlx::core::linalg::eigvalsh(arr_cpu, uplo, cpu_stream);
  array vals_target = astype(vals_cpu, vals_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(vals_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_eigh(SEXP xp_, std::string uplo, std::string device_str) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array arr_cpu = astype(arr, arr.dtype(), cpu_stream);
  auto eig_pair = mlx::core::linalg::eigh(arr_cpu, uplo, cpu_stream);

  array values_target = astype(eig_pair.first, eig_pair.first.dtype(), target_device);
  array vectors_target = astype(eig_pair.second, eig_pair.second.dtype(), target_device);

  return List::create(
      Named("values") = make_mlx_xptr(std::move(values_target)),
      Named("vectors") = make_mlx_xptr(std::move(vectors_target)));
}

// [[Rcpp::export]]
SEXP cpp_mlx_solve_triangular(SEXP a_xp_, SEXP b_xp_, bool upper,
                              std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);
  MlxArrayWrapper* b_wrapper = get_mlx_wrapper(b_xp_);

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array a_cpu = astype(a_wrapper->get(), a_wrapper->get().dtype(), cpu_stream);
  array b_cpu = astype(b_wrapper->get(), b_wrapper->get().dtype(), cpu_stream);

  array result_cpu = mlx::core::linalg::solve_triangular(a_cpu, b_cpu, upper, cpu_stream);
  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_cross(SEXP a_xp_, SEXP b_xp_, int axis, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);
  MlxArrayWrapper* b_wrapper = get_mlx_wrapper(b_xp_);

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array a_cpu = astype(a_wrapper->get(), a_wrapper->get().dtype(), cpu_stream);
  array b_cpu = astype(b_wrapper->get(), b_wrapper->get().dtype(), cpu_stream);

  int axis_input = axis;
  if (axis_input >= 0) {
    axis_input -= 1;
  }
  int ax = normalize_axis(a_cpu, axis_input);
  array result_cpu = mlx::core::linalg::cross(a_cpu, b_cpu, ax, cpu_stream);
  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_trace(SEXP a_xp_, int offset, int axis1, int axis2, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array a_cpu = astype(a_wrapper->get(), a_wrapper->get().dtype(), cpu_stream);

  // Convert 1-indexed to 0-indexed
  int ax1 = axis1 - 1;
  int ax2 = axis2 - 1;

  ax1 = normalize_axis(a_cpu, ax1);
  ax2 = normalize_axis(a_cpu, ax2);

  array result_cpu = trace(a_cpu, offset, ax1, ax2, cpu_stream);
  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_diagonal(SEXP a_xp_, int offset, int axis1, int axis2, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array a_cpu = astype(a_wrapper->get(), a_wrapper->get().dtype(), cpu_stream);

  // Convert 1-indexed to 0-indexed
  int ax1 = axis1 - 1;
  int ax2 = axis2 - 1;

  ax1 = normalize_axis(a_cpu, ax1);
  ax2 = normalize_axis(a_cpu, ax2);

  array result_cpu = diagonal(a_cpu, offset, ax1, ax2, cpu_stream);
  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_diag(SEXP a_xp_, int k, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array a_cpu = astype(a_wrapper->get(), a_wrapper->get().dtype(), cpu_stream);
  array result_cpu = diag(a_cpu, k, cpu_stream);
  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_outer(SEXP a_xp_, SEXP b_xp_, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);
  MlxArrayWrapper* b_wrapper = get_mlx_wrapper(b_xp_);

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array a_cpu = astype(a_wrapper->get(), a_wrapper->get().dtype(), cpu_stream);
  array b_cpu = astype(b_wrapper->get(), b_wrapper->get().dtype(), cpu_stream);

  array result_cpu = outer(a_cpu, b_cpu, cpu_stream);
  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);
  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_inv(SEXP a_xp_, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array a_cpu = astype(a_wrapper->get(), a_wrapper->get().dtype(), cpu_stream);
  array result_cpu = linalg::inv(a_cpu, cpu_stream);
  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);

  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_tri_inv(SEXP a_xp_, bool upper, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array a_cpu = astype(a_wrapper->get(), a_wrapper->get().dtype(), cpu_stream);
  array result_cpu = linalg::tri_inv(a_cpu, upper, cpu_stream);
  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);

  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_cholesky_inv(SEXP a_xp_, bool upper, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array a_cpu = astype(a_wrapper->get(), a_wrapper->get().dtype(), cpu_stream);
  array result_cpu = linalg::cholesky_inv(a_cpu, upper, cpu_stream);
  array result_target = astype(result_cpu, result_cpu.dtype(), target_device);

  return make_mlx_xptr(std::move(result_target));
}

// [[Rcpp::export]]
SEXP cpp_mlx_lu(SEXP a_xp_, std::string device_str) {
  MlxArrayWrapper* a_wrapper = get_mlx_wrapper(a_xp_);

  StreamOrDevice cpu_stream = Device(Device::cpu);
  StreamOrDevice target_device = string_to_device(device_str);

  array a_cpu = astype(a_wrapper->get(), a_wrapper->get().dtype(), cpu_stream);

  // lu() returns a vector of arrays [P, L, U]
  auto lu_result = linalg::lu(a_cpu, cpu_stream);

  if (lu_result.size() != 3) {
    Rcpp::stop("Unexpected LU result size");
  }

  array p_cpu = lu_result[0];  // Pivot indices
  array l_cpu = lu_result[1];  // Lower triangular
  array u_cpu = lu_result[2];  // Upper triangular

  array p_target = astype(p_cpu, p_cpu.dtype(), target_device);
  array l_target = astype(l_cpu, l_cpu.dtype(), target_device);
  array u_target = astype(u_cpu, u_cpu.dtype(), target_device);

  return List::create(
    Named("p") = make_mlx_xptr(std::move(p_target)),
    Named("l") = make_mlx_xptr(std::move(l_target)),
    Named("u") = make_mlx_xptr(std::move(u_target))
  );
}

