# Package index

## Core Tensor API

- [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md) :
  Create MLX array from R object
- [`is.mlx()`](https://hughjonesd.github.io/Rmlx/reference/is.mlx.md) :
  Test if object is an MLX array
- [`Rmlx-package`](https://hughjonesd.github.io/Rmlx/reference/Rmlx-package.md)
  [`Rmlx`](https://hughjonesd.github.io/Rmlx/reference/Rmlx-package.md)
  : Rmlx: R Interface to Apple's MLX Arrays
- [`mlx-methods`](https://hughjonesd.github.io/Rmlx/reference/mlx-methods.md)
  : Base R generics with mlx methods
- [`mlx_dim()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dim.md)
  : Get dimensions helper
- [`mlx_dtype()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dtype.md)
  : Get data type helper
- [`mlx_eval()`](https://hughjonesd.github.io/Rmlx/reference/mlx_eval.md)
  : Force evaluation of lazy MLX operations
- [`` `[`( ``*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/mlx_subset.md)
  [`` `[<-`( ``*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/mlx_subset.md)
  : Subset MLX array
- [`dim(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/dim.mlx.md)
  : Get dimensions of MLX array
- [`` `dim<-`( ``*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/dim-set-.mlx.md)
  : Set dimensions of MLX array
- [`length(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/length.mlx.md)
  : Get length of MLX array
- [`print(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/print.mlx.md)
  : Print MLX array
- [`str(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/str.mlx.md)
  : Object structure for MLX array
- [`t(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/t.mlx.md)
  : Transpose of MLX matrix
- [`as.matrix(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/as.matrix.mlx.md)
  : Convert MLX array to R matrix/array
- [`as.array(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/as.array.mlx.md)
  : Convert MLX array to R array
- [`as.vector(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/as.vector.mlx.md)
  : Convert MLX array to R vector
- [`as.logical(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/as.logical.mlx.md)
  : Convert MLX array to logical vector
- [`as.double(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/as.double.mlx.md)
  [`as.numeric(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/as.double.mlx.md)
  : Convert MLX array to numeric vector

## Device & Execution

- [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  : Get or set default MLX device
- [`with_default_device()`](https://hughjonesd.github.io/Rmlx/reference/with_default_device.md)
  : Temporarily set the default MLX device
- [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md)
  [`mlx_default_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md)
  : MLX streams for asynchronous execution
- [`mlx_set_default_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_set_default_stream.md)
  : Set the default MLX stream
- [`mlx_synchronize()`](https://hughjonesd.github.io/Rmlx/reference/mlx_synchronize.md)
  : Synchronize MLX execution
- [`mlx_forward()`](https://hughjonesd.github.io/Rmlx/reference/mlx_forward.md)
  : Forward pass utility
- [`mlx_grad()`](https://hughjonesd.github.io/Rmlx/reference/mlx_grad.md)
  [`mlx_value_grad()`](https://hughjonesd.github.io/Rmlx/reference/mlx_grad.md)
  : Automatic differentiation for MLX functions
- [`mlx_stop_gradient()`](https://hughjonesd.github.io/Rmlx/reference/mlx_stop_gradient.md)
  : Stop gradient propagation through an mlx array
- [`mlx_compile()`](https://hughjonesd.github.io/Rmlx/reference/mlx_compile.md)
  : Compile an MLX Function for Optimized Execution
- [`mlx_disable_compile()`](https://hughjonesd.github.io/Rmlx/reference/mlx_compile_control.md)
  [`mlx_enable_compile()`](https://hughjonesd.github.io/Rmlx/reference/mlx_compile_control.md)
  : Control Global Compilation Behavior

## Creation & Randomness

- [`mlx_array()`](https://hughjonesd.github.io/Rmlx/reference/mlx_array.md)
  : Construct an MLX array from R data
- [`mlx_matrix()`](https://hughjonesd.github.io/Rmlx/reference/mlx_matrix.md)
  : Construct MLX matrices efficiently
- [`mlx_vector()`](https://hughjonesd.github.io/Rmlx/reference/mlx_vector.md)
  : Construct MLX vectors
- [`mlx_scalar()`](https://hughjonesd.github.io/Rmlx/reference/mlx_scalar.md)
  : Construct MLX scalars
- [`mlx_zeros()`](https://hughjonesd.github.io/Rmlx/reference/mlx_zeros.md)
  : Create arrays of zeros on MLX devices
- [`mlx_ones()`](https://hughjonesd.github.io/Rmlx/reference/mlx_ones.md)
  : Create arrays of ones on MLX devices
- [`mlx_zeros_like()`](https://hughjonesd.github.io/Rmlx/reference/mlx_zeros_like.md)
  : Zeros shaped like an existing mlx array
- [`mlx_ones_like()`](https://hughjonesd.github.io/Rmlx/reference/mlx_ones_like.md)
  : Ones shaped like an existing mlx array
- [`mlx_full()`](https://hughjonesd.github.io/Rmlx/reference/mlx_full.md)
  : Fill an mlx array with a constant value
- [`mlx_eye()`](https://hughjonesd.github.io/Rmlx/reference/mlx_eye.md)
  : Identity-like matrices on MLX devices
- [`mlx_identity()`](https://hughjonesd.github.io/Rmlx/reference/mlx_identity.md)
  : Identity matrices on MLX devices
- [`mlx_arange()`](https://hughjonesd.github.io/Rmlx/reference/mlx_arange.md)
  : Numerical ranges on MLX devices
- [`mlx_linspace()`](https://hughjonesd.github.io/Rmlx/reference/mlx_linspace.md)
  : Evenly spaced ranges on MLX devices
- [`mlx_rand_bernoulli()`](https://hughjonesd.github.io/Rmlx/reference/mlx_rand_bernoulli.md)
  : Sample Bernoulli random variables on mlx arrays
- [`mlx_rand_categorical()`](https://hughjonesd.github.io/Rmlx/reference/mlx_rand_categorical.md)
  : Sample from a categorical distribution on mlx arrays
- [`mlx_rand_gumbel()`](https://hughjonesd.github.io/Rmlx/reference/mlx_rand_gumbel.md)
  : Sample from the Gumbel distribution on mlx arrays
- [`mlx_rand_laplace()`](https://hughjonesd.github.io/Rmlx/reference/mlx_rand_laplace.md)
  : Sample from the Laplace distribution on mlx arrays
- [`mlx_rand_multivariate_normal()`](https://hughjonesd.github.io/Rmlx/reference/mlx_rand_multivariate_normal.md)
  : Sample from a multivariate normal distribution on mlx arrays
- [`mlx_rand_normal()`](https://hughjonesd.github.io/Rmlx/reference/mlx_rand_normal.md)
  : Sample from a normal distribution on mlx arrays
- [`mlx_rand_permutation()`](https://hughjonesd.github.io/Rmlx/reference/mlx_rand_permutation.md)
  : Generate random permutations on mlx arrays
- [`mlx_rand_randint()`](https://hughjonesd.github.io/Rmlx/reference/mlx_rand_randint.md)
  : Sample random integers on mlx arrays
- [`mlx_rand_truncated_normal()`](https://hughjonesd.github.io/Rmlx/reference/mlx_rand_truncated_normal.md)
  : Sample from a truncated normal distribution on mlx arrays
- [`mlx_rand_uniform()`](https://hughjonesd.github.io/Rmlx/reference/mlx_rand_uniform.md)
  : Sample from a uniform distribution on mlx arrays
- [`mlx_key()`](https://hughjonesd.github.io/Rmlx/reference/mlx_key.md)
  [`mlx_key_split()`](https://hughjonesd.github.io/Rmlx/reference/mlx_key.md)
  : Construct MLX random number generator keys
- [`mlx_key_bits()`](https://hughjonesd.github.io/Rmlx/reference/mlx_key_bits.md)
  : Generate raw random bits on MLX arrays

## Shape & Indexing

- [`mlx_reshape()`](https://hughjonesd.github.io/Rmlx/reference/mlx_reshape.md)
  : Reshape an mlx array
- [`mlx_stack()`](https://hughjonesd.github.io/Rmlx/reference/mlx_stack.md)
  : Stack mlx arrays along a new axis
- [`mlx_squeeze()`](https://hughjonesd.github.io/Rmlx/reference/mlx_squeeze.md)
  : Remove singleton dimensions
- [`mlx_expand_dims()`](https://hughjonesd.github.io/Rmlx/reference/mlx_expand_dims.md)
  : Insert singleton dimensions
- [`mlx_repeat()`](https://hughjonesd.github.io/Rmlx/reference/mlx_repeat.md)
  : Repeat array elements
- [`mlx_tile()`](https://hughjonesd.github.io/Rmlx/reference/mlx_tile.md)
  : Tile an array
- [`mlx_pad()`](https://hughjonesd.github.io/Rmlx/reference/mlx_pad.md)
  [`mlx_split()`](https://hughjonesd.github.io/Rmlx/reference/mlx_pad.md)
  : Pad or split mlx arrays
- [`mlx_roll()`](https://hughjonesd.github.io/Rmlx/reference/mlx_roll.md)
  : Roll array elements
- [`mlx_moveaxis()`](https://hughjonesd.github.io/Rmlx/reference/mlx_moveaxis.md)
  [`aperm(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/mlx_moveaxis.md)
  : Reorder mlx array axes
- [`mlx_contiguous()`](https://hughjonesd.github.io/Rmlx/reference/mlx_contiguous.md)
  : Ensure contiguous memory layout
- [`mlx_flatten()`](https://hughjonesd.github.io/Rmlx/reference/mlx_flatten.md)
  : Flatten axes of an mlx array
- [`mlx_swapaxes()`](https://hughjonesd.github.io/Rmlx/reference/mlx_swapaxes.md)
  : Swap two axes of an mlx array
- [`drop()`](https://hughjonesd.github.io/Rmlx/reference/drop.md) : Drop
  singleton dimensions
- [`row()`](https://hughjonesd.github.io/Rmlx/reference/row.md)
  [`col()`](https://hughjonesd.github.io/Rmlx/reference/row.md) : Row
  and column indices for mlx arrays
- [`asplit()`](https://hughjonesd.github.io/Rmlx/reference/asplit.md) :
  Split mlx arrays along a margin
- [`mlx_unflatten()`](https://hughjonesd.github.io/Rmlx/reference/mlx_unflatten.md)
  : Unflatten an axis into multiple axes
- [`mlx_meshgrid()`](https://hughjonesd.github.io/Rmlx/reference/mlx_meshgrid.md)
  : Construct coordinate arrays from input vectors
- [`mlx_broadcast_to()`](https://hughjonesd.github.io/Rmlx/reference/mlx_broadcast_to.md)
  : Broadcast an array to a new shape
- [`mlx_broadcast_arrays()`](https://hughjonesd.github.io/Rmlx/reference/mlx_broadcast_arrays.md)
  : Broadcast multiple arrays to a shared shape
- [`mlx_where()`](https://hughjonesd.github.io/Rmlx/reference/mlx_where.md)
  : Elementwise conditional selection
- [`mlx_tri()`](https://hughjonesd.github.io/Rmlx/reference/mlx_tri.md)
  [`mlx_tril()`](https://hughjonesd.github.io/Rmlx/reference/mlx_tri.md)
  [`mlx_triu()`](https://hughjonesd.github.io/Rmlx/reference/mlx_tri.md)
  : Triangular helpers for MLX arrays
- [`mlx_slice_update()`](https://hughjonesd.github.io/Rmlx/reference/mlx_slice_update.md)
  : Update a slice of an mlx array
- [`mlx_gather()`](https://hughjonesd.github.io/Rmlx/reference/mlx_gather.md)
  : Gather elements from an mlx array
- [`abind()`](https://hughjonesd.github.io/Rmlx/reference/abind.md) :
  Bind mlx arrays along an axis
- [`rbind(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/rbind.mlx.md)
  : Row-bind mlx arrays
- [`cbind(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/cbind.mlx.md)
  : Column-bind mlx arrays

## Ordering & Selection

- [`mlx_sort()`](https://hughjonesd.github.io/Rmlx/reference/mlx_sort.md)
  [`mlx_argsort()`](https://hughjonesd.github.io/Rmlx/reference/mlx_sort.md)
  : Sort and argsort for mlx arrays
- [`mlx_topk()`](https://hughjonesd.github.io/Rmlx/reference/mlx_topk.md)
  [`mlx_partition()`](https://hughjonesd.github.io/Rmlx/reference/mlx_topk.md)
  [`mlx_argpartition()`](https://hughjonesd.github.io/Rmlx/reference/mlx_topk.md)
  : Top-k selection and partitioning on mlx arrays
- [`mlx_argmax()`](https://hughjonesd.github.io/Rmlx/reference/mlx_argmax.md)
  [`mlx_argmin()`](https://hughjonesd.github.io/Rmlx/reference/mlx_argmax.md)
  : Argmax and argmin on mlx arrays

## Math & Reductions

- [`Math(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/Math.mlx.md)
  : Math operations for MLX arrays
- [`Ops(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/Ops.mlx.md)
  : Arithmetic and comparison operators for MLX arrays
- [`Summary(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/Summary.mlx.md)
  : Summary operations for MLX arrays
- [`mlx_sum()`](https://hughjonesd.github.io/Rmlx/reference/mlx_sum.md)
  [`mlx_prod()`](https://hughjonesd.github.io/Rmlx/reference/mlx_sum.md)
  [`mlx_all()`](https://hughjonesd.github.io/Rmlx/reference/mlx_sum.md)
  [`mlx_any()`](https://hughjonesd.github.io/Rmlx/reference/mlx_sum.md)
  [`mlx_mean()`](https://hughjonesd.github.io/Rmlx/reference/mlx_sum.md)
  [`mlx_var()`](https://hughjonesd.github.io/Rmlx/reference/mlx_sum.md)
  [`mlx_std()`](https://hughjonesd.github.io/Rmlx/reference/mlx_sum.md)
  : Reduce mlx arrays
- [`mean(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/mean.mlx.md)
  : Mean of MLX array elements
- [`mlx_cumsum()`](https://hughjonesd.github.io/Rmlx/reference/mlx_cumsum.md)
  [`mlx_cumprod()`](https://hughjonesd.github.io/Rmlx/reference/mlx_cumsum.md)
  : Cumulative sum and product
- [`mlx_quantile()`](https://hughjonesd.github.io/Rmlx/reference/mlx_quantile.md)
  [`quantile(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/mlx_quantile.md)
  : Compute quantiles of MLX arrays
- [`mlx_clip()`](https://hughjonesd.github.io/Rmlx/reference/mlx_clip.md)
  : Clip mlx array values into a range
- [`mlx_maximum()`](https://hughjonesd.github.io/Rmlx/reference/mlx_maximum.md)
  : Elementwise maximum of two mlx arrays
- [`mlx_minimum()`](https://hughjonesd.github.io/Rmlx/reference/mlx_minimum.md)
  : Elementwise minimum of two mlx arrays
- [`mlx_hadamard_transform()`](https://hughjonesd.github.io/Rmlx/reference/mlx_hadamard_transform.md)
  : Hadamard transform for MLX arrays
- [`mlx_softmax()`](https://hughjonesd.github.io/Rmlx/reference/mlx_softmax.md)
  : Softmax for mlx arrays
- [`mlx_logsumexp()`](https://hughjonesd.github.io/Rmlx/reference/mlx_logsumexp.md)
  : Log-sum-exp reduction for mlx arrays
- [`mlx_logcumsumexp()`](https://hughjonesd.github.io/Rmlx/reference/mlx_logcumsumexp.md)
  : Log cumulative sum exponential for mlx arrays
- [`mlx_isnan()`](https://hughjonesd.github.io/Rmlx/reference/mlx_isnan.md)
  [`mlx_isinf()`](https://hughjonesd.github.io/Rmlx/reference/mlx_isnan.md)
  [`mlx_isfinite()`](https://hughjonesd.github.io/Rmlx/reference/mlx_isnan.md)
  : Elementwise NaN and infinity predicates
- [`mlx_isposinf()`](https://hughjonesd.github.io/Rmlx/reference/mlx_isposinf.md)
  [`mlx_isneginf()`](https://hughjonesd.github.io/Rmlx/reference/mlx_isposinf.md)
  : Detect signed infinities in mlx arrays
- [`mlx_nan_to_num()`](https://hughjonesd.github.io/Rmlx/reference/mlx_nan_to_num.md)
  : Replace NaN and infinite values with finite numbers
- [`mlx_real()`](https://hughjonesd.github.io/Rmlx/reference/mlx_real.md)
  [`mlx_imag()`](https://hughjonesd.github.io/Rmlx/reference/mlx_real.md)
  [`mlx_conjugate()`](https://hughjonesd.github.io/Rmlx/reference/mlx_real.md)
  : Complex-valued helpers for mlx arrays
- [`mlx_degrees()`](https://hughjonesd.github.io/Rmlx/reference/mlx_degrees.md)
  [`mlx_radians()`](https://hughjonesd.github.io/Rmlx/reference/mlx_degrees.md)
  : Convert between radians and degrees
- [`mlx_erf()`](https://hughjonesd.github.io/Rmlx/reference/mlx_erf.md)
  [`mlx_erfinv()`](https://hughjonesd.github.io/Rmlx/reference/mlx_erf.md)
  : Error function and inverse error function
- [`mlx_isclose()`](https://hughjonesd.github.io/Rmlx/reference/mlx_isclose.md)
  : Element-wise approximate equality
- [`mlx_allclose()`](https://hughjonesd.github.io/Rmlx/reference/mlx_allclose.md)
  : Test if all elements of two arrays are close
- [`all.equal(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/all.equal.mlx.md)
  : Test if two MLX arrays are (nearly) equal
- [`colSums()`](https://hughjonesd.github.io/Rmlx/reference/colSums.md)
  : Column sums for mlx arrays
- [`rowSums()`](https://hughjonesd.github.io/Rmlx/reference/rowSums.md)
  : Row sums for mlx arrays
- [`colMeans()`](https://hughjonesd.github.io/Rmlx/reference/colMeans.md)
  : Column means for mlx arrays
- [`rowMeans()`](https://hughjonesd.github.io/Rmlx/reference/rowMeans.md)
  : Row means for mlx arrays
- [`scale(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/scale.mlx.md)
  : Scale mlx arrays
- [`fft()`](https://hughjonesd.github.io/Rmlx/reference/fft.md) : Fast
  Fourier Transform
- [`mlx_fft()`](https://hughjonesd.github.io/Rmlx/reference/mlx_fft.md)
  [`mlx_fft2()`](https://hughjonesd.github.io/Rmlx/reference/mlx_fft.md)
  [`mlx_fftn()`](https://hughjonesd.github.io/Rmlx/reference/mlx_fft.md)
  : Fast Fourier transforms for MLX arrays

## Probability Distributions

- [`mlx_dnorm()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dnorm.md)
  [`mlx_pnorm()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dnorm.md)
  [`mlx_qnorm()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dnorm.md)
  : Normal distribution functions
- [`mlx_dunif()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dunif.md)
  [`mlx_punif()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dunif.md)
  [`mlx_qunif()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dunif.md)
  : Uniform distribution functions
- [`mlx_dexp()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dexp.md)
  [`mlx_pexp()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dexp.md)
  [`mlx_qexp()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dexp.md)
  : Exponential distribution functions
- [`mlx_dlnorm()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dlnorm.md)
  [`mlx_plnorm()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dlnorm.md)
  [`mlx_qlnorm()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dlnorm.md)
  : Lognormal distribution functions
- [`mlx_dlogis()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dlogis.md)
  [`mlx_plogis()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dlogis.md)
  [`mlx_qlogis()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dlogis.md)
  : Logistic distribution functions

## Linear Algebra

- [`` `%*%`( ``*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/grapes-times-grapes-.mlx.md)
  : Matrix multiplication for MLX arrays
- [`mlx_addmm()`](https://hughjonesd.github.io/Rmlx/reference/mlx_addmm.md)
  : Fused matrix multiply and add for MLX arrays
- [`crossprod(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/crossprod.mlx.md)
  : Cross product
- [`tcrossprod(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/tcrossprod.mlx.md)
  : Transposed cross product
- [`outer()`](https://hughjonesd.github.io/Rmlx/reference/outer.md) :
  Outer product of two vectors
- [`diag()`](https://hughjonesd.github.io/Rmlx/reference/diag.md) :
  Diagonal matrix extraction and construction
- [`chol(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/chol.mlx.md)
  : Cholesky decomposition for mlx arrays
- [`chol2inv()`](https://hughjonesd.github.io/Rmlx/reference/chol2inv.md)
  : Inverse from Cholesky decomposition
- [`kronecker()`](https://hughjonesd.github.io/Rmlx/reference/kronecker.md)
  [`kronecker.default()`](https://hughjonesd.github.io/Rmlx/reference/kronecker.md)
  : Kronecker product dispatcher
- [`qr(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/qr.mlx.md)
  : QR decomposition for mlx arrays
- [`svd()`](https://hughjonesd.github.io/Rmlx/reference/svd.md) :
  Singular value decomposition
- [`svd(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/svd.mlx.md)
  : Singular value decomposition for mlx arrays
- [`solve(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/solve.mlx.md)
  : Solve a system of linear equations
- [`pinv()`](https://hughjonesd.github.io/Rmlx/reference/pinv.md) :
  Moore-Penrose pseudoinverse for MLX arrays
- [`mlx_kron()`](https://hughjonesd.github.io/Rmlx/reference/mlx_kron.md)
  : Kronecker product for mlx arrays
- [`mlx_inv()`](https://hughjonesd.github.io/Rmlx/reference/mlx_inv.md)
  : Compute matrix inverse
- [`mlx_tri_inv()`](https://hughjonesd.github.io/Rmlx/reference/mlx_tri_inv.md)
  : Compute triangular matrix inverse
- [`mlx_cholesky_inv()`](https://hughjonesd.github.io/Rmlx/reference/mlx_cholesky_inv.md)
  : Compute matrix inverse via Cholesky decomposition
- [`mlx_lu()`](https://hughjonesd.github.io/Rmlx/reference/mlx_lu.md) :
  LU factorization
- [`mlx_norm()`](https://hughjonesd.github.io/Rmlx/reference/mlx_norm.md)
  : Matrix and vector norms for mlx arrays
- [`mlx_solve_triangular()`](https://hughjonesd.github.io/Rmlx/reference/mlx_solve_triangular.md)
  [`backsolve()`](https://hughjonesd.github.io/Rmlx/reference/mlx_solve_triangular.md)
  : Solve triangular systems with mlx arrays
- [`mlx_trace()`](https://hughjonesd.github.io/Rmlx/reference/mlx_trace.md)
  : Matrix trace for mlx arrays
- [`diag(`*`<mlx>`*`)`](https://hughjonesd.github.io/Rmlx/reference/mlx_diagonal.md)
  [`mlx_diagonal()`](https://hughjonesd.github.io/Rmlx/reference/mlx_diagonal.md)
  : Extract diagonal or construct diagonal matrix for mlx arrays
- [`mlx_eig()`](https://hughjonesd.github.io/Rmlx/reference/mlx_eig.md)
  : Eigen decomposition for mlx arrays
- [`mlx_eigh()`](https://hughjonesd.github.io/Rmlx/reference/mlx_eigh.md)
  : Eigen decomposition of Hermitian mlx arrays
- [`mlx_eigvals()`](https://hughjonesd.github.io/Rmlx/reference/mlx_eigvals.md)
  : Eigenvalues of mlx arrays
- [`mlx_eigvalsh()`](https://hughjonesd.github.io/Rmlx/reference/mlx_eigvalsh.md)
  : Eigenvalues of Hermitian mlx arrays
- [`mlx_cross()`](https://hughjonesd.github.io/Rmlx/reference/mlx_cross.md)
  : Vector cross product with mlx arrays

## Input & Output

- [`mlx_save()`](https://hughjonesd.github.io/Rmlx/reference/mlx_save.md)
  : Save an MLX array to disk
- [`mlx_load()`](https://hughjonesd.github.io/Rmlx/reference/mlx_load.md)
  : Load an MLX array from disk
- [`mlx_save_safetensors()`](https://hughjonesd.github.io/Rmlx/reference/mlx_save_safetensors.md)
  : Save MLX arrays to the safetensors format
- [`mlx_load_safetensors()`](https://hughjonesd.github.io/Rmlx/reference/mlx_load_safetensors.md)
  : Load MLX arrays from the safetensors format
- [`mlx_save_gguf()`](https://hughjonesd.github.io/Rmlx/reference/mlx_save_gguf.md)
  : Save MLX arrays to the GGUF format
- [`mlx_load_gguf()`](https://hughjonesd.github.io/Rmlx/reference/mlx_load_gguf.md)
  : Load MLX tensors from the GGUF format
- [`mlx_import_function()`](https://hughjonesd.github.io/Rmlx/reference/mlx_import_function.md)
  : Import an exported MLX function

## Neural Network Layers

- [`mlx_linear()`](https://hughjonesd.github.io/Rmlx/reference/mlx_linear.md)
  : Create a learnable linear transformation
- [`mlx_sequential()`](https://hughjonesd.github.io/Rmlx/reference/mlx_sequential.md)
  : Compose modules sequentially
- [`mlx_set_training()`](https://hughjonesd.github.io/Rmlx/reference/mlx_set_training.md)
  : Toggle training mode for MLX modules
- [`mlx_embedding()`](https://hughjonesd.github.io/Rmlx/reference/mlx_embedding.md)
  : Embedding layer
- [`mlx_conv1d()`](https://hughjonesd.github.io/Rmlx/reference/mlx_conv1d.md)
  : 1D Convolution
- [`mlx_conv2d()`](https://hughjonesd.github.io/Rmlx/reference/mlx_conv2d.md)
  : 2D Convolution
- [`mlx_conv3d()`](https://hughjonesd.github.io/Rmlx/reference/mlx_conv3d.md)
  : 3D Convolution
- [`mlx_conv_transpose1d()`](https://hughjonesd.github.io/Rmlx/reference/mlx_conv_transpose1d.md)
  : 1D Transposed Convolution
- [`mlx_conv_transpose2d()`](https://hughjonesd.github.io/Rmlx/reference/mlx_conv_transpose2d.md)
  : 2D Transposed Convolution
- [`mlx_conv_transpose3d()`](https://hughjonesd.github.io/Rmlx/reference/mlx_conv_transpose3d.md)
  : 3D Transposed Convolution
- [`mlx_quantize()`](https://hughjonesd.github.io/Rmlx/reference/mlx_quantize.md)
  : Quantize a Matrix
- [`mlx_dequantize()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dequantize.md)
  : Dequantize a Matrix
- [`mlx_quantized_matmul()`](https://hughjonesd.github.io/Rmlx/reference/mlx_quantized_matmul.md)
  : Quantized Matrix Multiplication
- [`mlx_gather_qmm()`](https://hughjonesd.github.io/Rmlx/reference/mlx_gather_qmm.md)
  : Gather-based Quantized Matrix Multiplication

## Activation Functions

- [`mlx_relu()`](https://hughjonesd.github.io/Rmlx/reference/mlx_relu.md)
  : Rectified linear activation module
- [`mlx_gelu()`](https://hughjonesd.github.io/Rmlx/reference/mlx_gelu.md)
  : GELU activation
- [`mlx_sigmoid()`](https://hughjonesd.github.io/Rmlx/reference/mlx_sigmoid.md)
  : Sigmoid activation
- [`mlx_tanh()`](https://hughjonesd.github.io/Rmlx/reference/mlx_tanh.md)
  : Tanh activation
- [`mlx_silu()`](https://hughjonesd.github.io/Rmlx/reference/mlx_silu.md)
  : SiLU (Swish) activation
- [`mlx_leaky_relu()`](https://hughjonesd.github.io/Rmlx/reference/mlx_leaky_relu.md)
  : Leaky ReLU activation
- [`mlx_softmax_layer()`](https://hughjonesd.github.io/Rmlx/reference/mlx_softmax_layer.md)
  : Softmax activation

## Regularization & Normalization

- [`mlx_dropout()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dropout.md)
  : Dropout layer
- [`mlx_layer_norm()`](https://hughjonesd.github.io/Rmlx/reference/mlx_layer_norm.md)
  : Layer normalization
- [`mlx_batch_norm()`](https://hughjonesd.github.io/Rmlx/reference/mlx_batch_norm.md)
  : Batch normalization

## Loss Functions

- [`mlx_mse_loss()`](https://hughjonesd.github.io/Rmlx/reference/mlx_mse_loss.md)
  : Mean squared error loss
- [`mlx_l1_loss()`](https://hughjonesd.github.io/Rmlx/reference/mlx_l1_loss.md)
  : L1 loss (Mean Absolute Error)
- [`mlx_binary_cross_entropy()`](https://hughjonesd.github.io/Rmlx/reference/mlx_binary_cross_entropy.md)
  : Binary cross-entropy loss
- [`mlx_cross_entropy()`](https://hughjonesd.github.io/Rmlx/reference/mlx_cross_entropy.md)
  : Cross-entropy loss

## Training Utilities

- [`mlx_parameters()`](https://hughjonesd.github.io/Rmlx/reference/mlx_parameters.md)
  : Collect parameters from modules
- [`mlx_param_values()`](https://hughjonesd.github.io/Rmlx/reference/mlx_param_values.md)
  : Retrieve parameter arrays
- [`mlx_param_set_values()`](https://hughjonesd.github.io/Rmlx/reference/mlx_param_set_values.md)
  : Assign arrays back to parameters
- [`mlx_optimizer_sgd()`](https://hughjonesd.github.io/Rmlx/reference/mlx_optimizer_sgd.md)
  : Stochastic gradient descent optimizer
- [`mlx_train_step()`](https://hughjonesd.github.io/Rmlx/reference/mlx_train_step.md)
  : Single training step helper
- [`mlx_coordinate_descent()`](https://hughjonesd.github.io/Rmlx/reference/mlx_coordinate_descent.md)
  : Coordinate Descent with L1 Regularization
