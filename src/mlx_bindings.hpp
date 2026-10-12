#ifndef MLX_BINDINGS_HPP
#define MLX_BINDINGS_HPP

#include <Rcpp.h>
#include <complex>
#include <mlx/mlx.h>
#include <string>
#include <utility>
#include <mlx/export.h>

namespace rmlx {

class MlxStreamWrapper {
private:
  mlx::core::Stream stream_;

public:
  explicit MlxStreamWrapper(mlx::core::Stream stream) : stream_(stream) {}
  ~MlxStreamWrapper() = default;

  MlxStreamWrapper(const MlxStreamWrapper&) = delete;
  MlxStreamWrapper& operator=(const MlxStreamWrapper&) = delete;

  MlxStreamWrapper(MlxStreamWrapper&&) = delete;
  MlxStreamWrapper& operator=(MlxStreamWrapper&&) = delete;

  const mlx::core::Stream& get() const { return stream_; }
};

class MlxImportedFunctionWrapper {
private:
  mlx::core::ImportedFunction function_;

public:
  explicit MlxImportedFunctionWrapper(mlx::core::ImportedFunction function)
      : function_(std::move(function)) {}
  ~MlxImportedFunctionWrapper() = default;

  const mlx::core::ImportedFunction& get() const { return function_; }
};

// Finalizer for R external pointers
void mlx_array_finalizer(SEXP xp);
void mlx_stream_finalizer(SEXP xp);
void mlx_imported_function_finalizer(SEXP xp);

// Get the native array after validating its external pointer.
mlx::core::array& get_mlx_array(SEXP xp);
mlx::core::Stream get_mlx_stream(SEXP xp);
class MlxImportedFunctionWrapper;
MlxImportedFunctionWrapper* get_mlx_imported_function(SEXP xp);

// Helper to wrap MLX array in external pointer
SEXP make_mlx_xptr(const mlx::core::array& arr);
SEXP make_mlx_xptr(mlx::core::array&& arr);
SEXP make_mlx_stream_xptr(mlx::core::Stream stream);
SEXP make_mlx_imported_function_xptr(mlx::core::ImportedFunction function);

// Helper: convert dtype string to MLX dtype
mlx::core::Dtype string_to_dtype(const std::string& dtype);

// Helper: convert dtype to string
std::string dtype_to_string(mlx::core::Dtype dtype);

// Helper: convert device string to MLX device
mlx::core::Device string_to_device(const std::string& device);
std::string device_to_string(const mlx::core::Device& device);

// Helper: determine if SEXP is stream external pointer
bool is_mlx_stream(SEXP value);

} // namespace rmlx

#endif // MLX_BINDINGS_HPP
