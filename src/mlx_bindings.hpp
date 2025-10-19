#ifndef MLX_BINDINGS_HPP
#define MLX_BINDINGS_HPP

#include <Rcpp.h>
#include <mlx/mlx.h>
#include <memory>

namespace rmlx {

// Wrapper for MLX array using shared_ptr
class MlxArrayWrapper {
private:
  std::shared_ptr<mlx::core::array> ptr_;

public:
  MlxArrayWrapper();
  explicit MlxArrayWrapper(const mlx::core::array& arr);
  explicit MlxArrayWrapper(mlx::core::array&& arr);
  ~MlxArrayWrapper() = default;

  // Delete copy operations
  MlxArrayWrapper(const MlxArrayWrapper&) = delete;
  MlxArrayWrapper& operator=(const MlxArrayWrapper&) = delete;

  // Move operations
  MlxArrayWrapper(MlxArrayWrapper&& other) noexcept = default;
  MlxArrayWrapper& operator=(MlxArrayWrapper&& other) noexcept = default;

  // Accessors
  mlx::core::array& get() { return *ptr_; }
  const mlx::core::array& get() const { return *ptr_; }

  bool is_null() const { return ptr_ == nullptr; }
};

// Finalizer for R external pointers
void mlx_array_finalizer(SEXP xp);

// Helper to unwrap external pointer to MlxArrayWrapper
MlxArrayWrapper* get_mlx_wrapper(SEXP xp);

// Helper to wrap MLX array in external pointer
SEXP make_mlx_xptr(const mlx::core::array& arr);
SEXP make_mlx_xptr(mlx::core::array&& arr);

// Helper: convert dtype string to MLX dtype
mlx::core::Dtype string_to_dtype(const std::string& dtype);

// Helper: convert dtype to string
std::string dtype_to_string(mlx::core::Dtype dtype);

// Helper: convert device string to MLX device
mlx::core::Device string_to_device(const std::string& device);

} // namespace rmlx

#endif // MLX_BINDINGS_HPP
