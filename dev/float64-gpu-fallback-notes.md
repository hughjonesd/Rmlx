# Float64 GPU fallback diagnostic notes

These notes cover the 25 diagnostic "successes" from
`tests/testthat/test-diagnostic-float64-gpu-fallbacks.R` on the
`diagnose-float64-gpu-fallbacks` branch. In this branch, float64 device
validation is intentionally removed, so a success means that a GPU-tagged
float64 input did not fail when the operation was evaluated.

The important distinction is that a success does not always mean "MLX moved the
operation to CPU". In most cases below, Rmlx itself explicitly casts inputs to a
CPU stream, calls MLX on that CPU stream, and then casts the result back to the
Rmlx object's preferred device tag. In a few cases, the operation likely does no
array computation for the tested input.

Local MLX headers checked:

- `/opt/homebrew/include/mlx/linalg.h`
- `/opt/homebrew/include/mlx/fft.h`
- `/opt/homebrew/include/mlx/ops.h`
- `/opt/homebrew/include/mlx/backend/metal/kernels/fft.h`

## Current branch update

The corrected diagnostic now keeps MLX's global default device on CPU while
creating GPU-tagged float64 inputs. That distinguishes operations that respect
the Rmlx object device from operations that merely follow MLX's global default.

The first corrected run found 83 operations that succeeded because wrappers were
not passing `x$device` to MLX. Those were fixed by passing explicit
`StreamOrDevice` values through the shared wrapper families for unary math,
reductions, sorting, shape transforms, indexing/slicing, cumulative operations,
and column-major boundary helpers. After that change, the only successes are
identity/no-op style cases for this diagnostic input: `mlx_cast`,
`mlx_conjugate`, `mlx_imag`, and `mlx_real`.

The explicit CPU staging has now been removed for the operations below. A GPU
float32 probe on MLX 0.31.1 produced these results:

- Now runs on GPU: `diag.mlx`, `fft.mlx`, `mlx_cross`, `mlx_diagonal`,
  `mlx_fft`, `mlx_fft2`, `mlx_fftn`, `mlx_norm`, `mlx_trace`,
  `mlx_unflatten`, `outer.mlx`.
- Now respects GPU and lets MLX report unsupported execution: `chol.mlx`,
  `chol2inv`, `mlx_cholesky_inv`, `mlx_eig`, `mlx_eigh`, `mlx_eigvals`,
  `mlx_eigvalsh`, `mlx_inv`, `mlx_lu`, `mlx_solve_triangular`,
  `mlx_tri_inv`, `pinv`, `qr.mlx`, `solve.mlx`, `svd`, `svd.mlx`.
- Likely no-op or identity-style cases in the original diagnostic:
  `mlx_cast`, `mlx_conjugate`, `mlx_imag`, `mlx_real`.

A later source scan found two more wrappers with the same explicit CPU staging
pattern. After removing it, `mlx_cross()` runs on GPU and
`mlx_solve_triangular()` respects GPU but MLX 0.31.1 reports that triangular
solves are not yet supported there.

The notes below describe the original diagnostic state and the specific source
patterns that caused each success before these wrappers were changed.

## chol.mlx

- Underlying MLX capability: `mlx::core::linalg::cholesky()` accepts a
  `StreamOrDevice` in `mlx/linalg.h`, so the public MLX API is
  stream/device-parameterized. With MLX 0.31.1, attempting Cholesky on GPU
  errors: "This op is not yet supported on the GPU."
- Previous reason this succeeded: `chol.mlx()` passed `x$device` to
  `cpp_mlx_cholesky()`, but `src/mlx_linalg.cpp` cast the input to
  `Device(Device::cpu)`, called `linalg::cholesky(..., cpu_stream)`, then cast
  the result back to the target device.
- Current branch status: `cpp_mlx_cholesky()` now casts to
  `typed_device(target_dtype, device_str)` and calls
  `linalg::cholesky(..., target_device)`, so it respects the operand's
  preferred device. CPU operands run; GPU operands receive the MLX unsupported
  GPU error instead of silently falling back to CPU.
- Documentation: no Rmlx CPU-only note found in the roxygen/Rd for `chol.mlx`;
  that now matches the implementation.

## chol2inv

- Underlying MLX capability: Rmlx implements this through
  `mlx::core::linalg::cholesky_inv()`, which accepts `StreamOrDevice` in
  `mlx/linalg.h`.
- Why this succeeds: `chol2inv.mlx()` calls `mlx_cholesky_inv()`, whose C++
  wrapper casts to CPU, calls `linalg::cholesky_inv(..., cpu_stream)`, then
  casts the result back to `x$device`.
- Documentation: no Rmlx CPU-only note found for `chol2inv` or
  `mlx_cholesky_inv`.

## diag.mlx

- Underlying MLX capability: `mlx::core::diag()` accepts `StreamOrDevice` in
  `mlx/ops.h`.
- Why this succeeds: `diag.mlx()` calls `cpp_mlx_diag(..., x$device)`, but
  `cpp_mlx_diag()` casts the input to CPU, calls `diag(..., cpu_stream)`, then
  casts the result back to the target device.
- Documentation: no Rmlx CPU-only note found for `diag.mlx`.

## fft.mlx

- Underlying MLX capability: MLX FFT functions accept `StreamOrDevice` in
  `mlx/fft.h`, and local headers include Metal FFT kernels, so FFT is not merely
  a CPU-only API for supported dtypes.
- Why this succeeds: `fft.mlx()` dispatches to `cpp_mlx_fft()`. The wrapper
  casts the input to a CPU stream, calls `fftn()` or `ifftn()` on that CPU
  stream, then casts the result back to the requested target device.
- Documentation: Rmlx documentation says that when `device = NULL`, the
  transform runs on the input array's device. That does not match the current
  C++ implementation.

## mlx_cast

- Underlying MLX capability: `astype()` accepts `StreamOrDevice` in `mlx/ops.h`.
- Why this succeeds: this diagnostic uses a same-dtype, same-device cast from
  GPU-tagged float64 to GPU-tagged float64. The Rmlx wrapper calls
  `astype(arr, dtype, dev)`, but this case likely simplifies to an identity or
  lazy metadata-preserving result and does not prove that a float64 GPU kernel
  ran or that Rmlx fell back to CPU.
- Documentation: no CPU-only note is needed for this specific no-op result, but
  the diagnostic does not establish real GPU float64 cast support.

## mlx_cholesky_inv

- Underlying MLX capability: `mlx::core::linalg::cholesky_inv()` accepts
  `StreamOrDevice` in `mlx/linalg.h`.
- Why this succeeds: `cpp_mlx_cholesky_inv()` casts the input to CPU, calls
  `linalg::cholesky_inv(..., cpu_stream)`, then casts the result back to the
  target device.
- Documentation: no Rmlx CPU-only note found.

## mlx_conjugate

- Underlying MLX capability: `mlx::core::conjugate()` accepts `StreamOrDevice`
  in `mlx/ops.h`.
- Why this succeeds: Rmlx calls `conjugate(wrapper->get())` through the generic
  unary wrapper without passing the object's preferred device. In this
  diagnostic the input is real float64, so conjugate is expected to be an
  identity-style operation. This success does not show a CPU fallback; it likely
  shows that no real computation was needed for a real-valued input.
- Documentation: no Rmlx CPU-only note found.

## mlx_diagonal

- Underlying MLX capability: `mlx::core::diagonal()` accepts `StreamOrDevice`
  in `mlx/ops.h`.
- Why this succeeds: `cpp_mlx_diagonal()` casts the input to CPU, calls
  `diagonal(..., cpu_stream)`, then casts the result back to the target device.
- Documentation: no Rmlx CPU-only note found.

## mlx_eig

- Underlying MLX capability: `mlx::core::linalg::eig()` accepts
  `StreamOrDevice` in `mlx/linalg.h`.
- Why this succeeds: `cpp_mlx_eig()` casts the input to CPU, calls
  `linalg::eig(..., cpu_stream)`, then casts both values and vectors back to the
  target device.
- Documentation: no Rmlx CPU-only note found.

## mlx_eigh

- Underlying MLX capability: `mlx::core::linalg::eigh()` accepts
  `StreamOrDevice` in `mlx/linalg.h`.
- Why this succeeds: `cpp_mlx_eigh()` casts the input to CPU, calls
  `linalg::eigh(..., cpu_stream)`, then casts both values and vectors back to
  the target device.
- Documentation: no Rmlx CPU-only note found.

## mlx_eigvals

- Underlying MLX capability: `mlx::core::linalg::eigvals()` accepts
  `StreamOrDevice` in `mlx/linalg.h`.
- Why this succeeds: `cpp_mlx_eigvals()` casts the input to CPU, calls
  `linalg::eigvals(..., cpu_stream)`, then casts the result back to the target
  device.
- Documentation: no Rmlx CPU-only note found.

## mlx_eigvalsh

- Underlying MLX capability: `mlx::core::linalg::eigvalsh()` accepts
  `StreamOrDevice` in `mlx/linalg.h`.
- Why this succeeds: `cpp_mlx_eigvalsh()` casts the input to CPU, calls
  `linalg::eigvalsh(..., cpu_stream)`, then casts the result back to the target
  device.
- Documentation: no Rmlx CPU-only note found.

## mlx_fft

- Underlying MLX capability: MLX `fft()` is an inline wrapper over `fftn()`,
  and the FFT API accepts `StreamOrDevice`; local Metal FFT kernels are present.
- Why this succeeds: `mlx_fft()` dispatches to `cpp_mlx_fft()`, which casts the
  input to CPU, calls MLX FFT on a CPU stream, and casts the result back to the
  requested device.
- Documentation: Rmlx documentation says the transform runs on the input
  array's device when `device = NULL`; this is not what the wrapper currently
  does.

## mlx_fft2

- Underlying MLX capability: MLX `fft2()` is an inline wrapper over `fftn()`,
  and the FFT API accepts `StreamOrDevice`; local Metal FFT kernels are present.
- Why this succeeds: `mlx_fft2()` dispatches to the same `cpp_mlx_fft()` CPU
  path as `mlx_fft()`.
- Documentation: the same FFT documentation mismatch applies.

## mlx_fftn

- Underlying MLX capability: MLX `fftn()` accepts `StreamOrDevice` directly in
  `mlx/fft.h`; local Metal FFT kernels are present.
- Why this succeeds: `mlx_fftn()` dispatches to `cpp_mlx_fft()`, which always
  performs the FFT call on a CPU stream before casting the result to the target
  device.
- Documentation: the same FFT documentation mismatch applies.

## mlx_inv

- Underlying MLX capability: `mlx::core::linalg::inv()` accepts
  `StreamOrDevice` in `mlx/linalg.h`, but Rmlx already documents this operation
  as CPU-run.
- Why this succeeds: `cpp_mlx_inv()` casts the input to CPU, calls
  `linalg::inv(..., cpu_stream)`, then casts the result back to the target
  device.
- Documentation: Rmlx roxygen/Rd for `mlx_inv()` says that, as of MLX 0.30.0,
  this runs on the CPU. This is the one success in this set with explicit Rmlx
  CPU documentation.

## mlx_lu

- Underlying MLX capability: `mlx::core::linalg::lu()` accepts
  `StreamOrDevice` in `mlx/linalg.h`.
- Why this succeeds: `cpp_mlx_lu()` casts the input to CPU, calls
  `linalg::lu(..., cpu_stream)`, then casts `p`, `l`, and `u` back to the target
  device.
- Documentation: no Rmlx CPU-only note found.

## mlx_norm

- Underlying MLX capability: the overloads of `mlx::core::linalg::norm()`
  accept `StreamOrDevice` in `mlx/linalg.h`.
- Why this succeeds: `cpp_mlx_norm()` casts the input to CPU, calls the selected
  `linalg::norm(..., cpu_stream)` overload, then casts the result back to the
  target device.
- Documentation: no Rmlx CPU-only note found.

## mlx_real

- Underlying MLX capability: `mlx::core::real()` accepts `StreamOrDevice` in
  `mlx/ops.h`.
- Why this succeeds: Rmlx calls `real(wrapper->get())` through the generic unary
  wrapper without passing the object's preferred device. In this diagnostic the
  input is already real float64, so `real()` is expected to be an identity-style
  operation. This success does not show a CPU fallback.
- Documentation: no Rmlx CPU-only note found.

## mlx_trace

- Underlying MLX capability: `mlx::core::trace()` accepts `StreamOrDevice` in
  `mlx/ops.h`.
- Why this succeeds: `cpp_mlx_trace()` casts the input to CPU, calls
  `trace(..., cpu_stream)`, then casts the result back to the target device.
- Documentation: no Rmlx CPU-only note found.

## mlx_tri_inv

- Underlying MLX capability: `mlx::core::linalg::tri_inv()` accepts
  `StreamOrDevice` in `mlx/linalg.h`.
- Why this succeeds: `cpp_mlx_tri_inv()` casts the input to CPU, calls
  `linalg::tri_inv(..., cpu_stream)`, then casts the result back to the target
  device.
- Documentation: Rmlx documentation warns that MLX may crash if the input is not
  triangular, but it does not say the operation is CPU-only.

## mlx_unflatten

- Underlying MLX capability: `mlx::core::unflatten()` accepts `StreamOrDevice`
  in `mlx/ops.h`.
- Why this succeeds: `cpp_mlx_unflatten()` casts the input to CPU, normalizes
  the axis against the CPU array, calls `unflatten(..., cpu_stream)`, then casts
  the result back to the target device.
- Documentation: no Rmlx CPU-only note found.

## outer.mlx

- Underlying MLX capability: `mlx::core::outer()` accepts `StreamOrDevice` in
  `mlx/ops.h`.
- Why this succeeds: `outer.mlx()` calls `cpp_mlx_outer()`, which casts both
  operands to CPU, calls `outer(..., cpu_stream)`, then casts the result back to
  the target device.
- Documentation: no Rmlx CPU-only note found.

## pinv

- Underlying MLX capability: `mlx::core::linalg::pinv()` accepts
  `StreamOrDevice` in `mlx/linalg.h`.
- Why this succeeds: `pinv()` calls `cpp_mlx_pinv()`, which casts the input to
  CPU, calls `linalg::pinv(..., cpu_stream)`, then casts the result back to the
  target device.
- Documentation: no Rmlx CPU-only note found.

## qr.mlx

- Underlying MLX capability: `mlx::core::linalg::qr()` accepts
  `StreamOrDevice` in `mlx/linalg.h`.
- Why this succeeds: `qr.mlx()` calls `cpp_mlx_qr()`, which casts the input to
  CPU, calls `linalg::qr(..., cpu_stream)`, then casts both `Q` and `R` back to
  the target device.
- Documentation: no Rmlx CPU-only note found.
