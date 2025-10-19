

# Project: `Rmlx`

## High-level scope (for the agent)

* Language: R (+ C++ via Rcpp).
* OS: macOS on Apple Silicon (Metal backend). Cross-platform explicitly **out of scope** now.
* Dependency: Apple MLX C/C++ library (present on system). We’ll **not** vendor MLX; we’ll detect and link to it.
* Core object: S3 class `mlx` wrapping an external pointer to an MLX array.
* Semantics: **lazy** by default. `as.matrix.mlx()` (and some other conversions) **force evaluation**; otherwise, users call `mlx_eval(x)`.
* Operator overloading: arithmetic, matrix algebra, comparisons; plus targeted stats helpers (`colMeans.mlx`, `rowMeans.mlx`, `crossprod.mlx`, `tcrossprod.mlx`, `t.mlx`, `sum.mlx`, `mean.mlx`).
* Phase 1: low-level arrays + ops, evaluation, conversions, basic reductions, tests, doc.
  (Autodiff/optimizers reserved for a later phase, not implemented now.)

---

## Milestones

1. **Scaffold & toolchain**
2. **C++ core: array handle, create/convert/eval**
3. **Binary ops & reductions**
4. **Matrix algebra & helpers**
5. **R S3 class + operator overloading**
6. **Indexing, printing, and diagnostics**
7. **Device/stream management**
8. **Docs, examples, tests**
9. **Build, local check, packaging**

Each milestone below lists atomic tasks.

---

## Milestone 1 — Scaffold & toolchain

**T1. Create package skeleton**

* Create standard R package layout:

  ```
  Rmlx/
    R/
    src/
    inst/
    man/
    tests/testthat/
    vignettes/
    DESCRIPTION
    NAMESPACE
    .Rbuildignore
    .gitignore
    configure     # POSIX shell
    cleanup       # optional
  ```
* Acceptance: `R CMD build` produces a tarball; `R CMD check` runs (expect skips for missing MLX until T3/T4 add detection).

**T2. DESCRIPTION**

* Minimal fields:

  * `Package: Rmlx`
  * `Type: Package`
  * `Title: R Interface to Apple's MLX Arrays (GPU-Accelerated on Apple Silicon)`
  * `Version: 0.0.0.9000`
  * `Authors@R: person("First","Last", role=c("aut","cre"), email="you@example.com")`
  * `Description: S3 class 'mlx' backed by Apple MLX arrays with lazy GPU ops via Rcpp.`
  * `License: MIT + file LICENSE`
  * `Encoding: UTF-8`
  * `Depends: R (>= 4.1.0)`
  * `Imports: Rcpp (>= 1.0.10)`
  * `LinkingTo: Rcpp`
  * `Suggests: testthat (>= 3.0.0), knitr, rmarkdown`
  * `Config/testthat/edition: 3`
  * `SystemRequirements: MLX (Apple Machine Learning eXchange) with C/C++ headers and library; macOS on Apple Silicon`
* Acceptance: `R CMD check` reads DESCRIPTION cleanly.

**T3. NAMESPACE**

* Start with:

  ```
  useDynLib(Rmlx, .registration = TRUE)
  importFrom(Rcpp, sourceCpp)
  S3method(print, mlx)
  S3method(as.matrix, mlx)
  S3method(colMeans, mlx)
  S3method(rowMeans, mlx)
  S3method(t, mlx)
  S3method(sum, mlx)
  S3method(mean, mlx)
  S3method(crossprod, mlx)
  S3method(tcrossprod, mlx)
  S3method(Ops, mlx)
  S3method(MatMult, mlx)       # custom for %*% (see T17)
  ```
* We’ll also register `%*%` for `mlx` via `setMethod` pattern in R (see T17).

**T4. Build-time MLX detection (`configure`)**

* Implement POSIX shell `configure` that:

  * Looks for MLX headers & libs (probe typical locations: `/opt/homebrew/include`, `/opt/homebrew/lib`, `/usr/local/include`, `/usr/local/lib`, `xcrun -sdk macosx --show-sdk-path`).
  * Allows env overrides:

    * `MLX_INCLUDE`, `MLX_LIB_DIR`, `MLX_LIBS` (e.g., `-lmlx -lc++`)
  * Writes `src/Makevars` with `PKG_CPPFLAGS` including `-I...` and `PKG_LIBS` including `-L... -lmlx`.
* Add a **helpful error** if not found: write a small header check and fail with a message telling the user to install MLX (Homebrew tap or Apple docs).
* Acceptance: `R CMD INSTALL .` fails gracefully if MLX missing; succeeds when include+lib provided.

> Gotcha: keep `Makevars` minimal and platform-guarded; do not hardcode Intel paths.

---

## Milestone 2 — C++ core: array handle, create/convert/eval

**T5. Define C++ wrapper class & finalizer**

* File: `src/mlx_bindings.cpp`, plus a header `src/mlx_bindings.hpp`.
* Create a thin RAII wrapper around MLX array handle (consult MLX C API names; assume types like `mlx_array*`):

  ```cpp
  struct MlxArray {
    mlx_array* ptr;
    MlxArray();                       // null
    explicit MlxArray(mlx_array* p);  // takes ownership
    ~MlxArray();                      // calls mlx_array_free(ptr)
    MlxArray(const MlxArray&) = delete;
    MlxArray& operator=(const MlxArray&) = delete;
    MlxArray(MlxArray&&) noexcept;
    MlxArray& operator=(MlxArray&&) noexcept;
  };
  ```
* Expose as external pointer to R with a finalizer that deletes the `MlxArray`.
* Acceptance: Creating and garbage-collecting an `mlx` object does not leak (use valgrind locally if possible).

**T6. Create MLX array from R data**

* Rcpp-exposed functions (C++):

  * `SEXP cpp_mlx_from_numeric(SEXP x, SEXP dim, SEXP dtype, SEXP device);`

    * Inputs: `x` as `NumericVector` (contiguous), `dim` as `IntegerVector`, `dtype` string ("float32"/"float64"), `device` string ("gpu"/"cpu").
    * Create MLX array with given shape and copy host data.
  * `SEXP cpp_mlx_empty(SEXP dim, SEXP dtype, SEXP device);`
* Acceptance: `as_mlx(matrix(...))` returns an `mlx` with matching shape and dtype.

**T7. Copy MLX array back to R**

* Function:

  * `SEXP cpp_mlx_to_numeric(SEXP x);`

    * Ensure **evaluation** first (see T8). Then copy to `NumericVector` (column-major like R).
* Acceptance: `as.matrix.mlx(x)` yields identical values to input after a roundtrip.

**T8. Evaluation**

* Implement:

  * `void cpp_mlx_eval(SEXP x);` (force compute & sync).
* Design: store a “needs_eval” flag inside the wrapper, or rely on MLX’s own state; call MLX `eval` on the array or current graph root.
* R side:

  * `mlx_eval(x)` calls into `cpp_mlx_eval`.
  * `as.matrix.mlx()` **must** call `mlx_eval()` before copying.
* Acceptance: Composed lazy ops only compute once evaluated or converted.

---

## Milestone 3 — Binary ops & reductions

**T9. Elementwise unary ops**

* C++ functions: `neg`, `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, etc.
* Signature pattern:

  * `SEXP cpp_mlx_unary(SEXP x, std::string op); // op ∈ {"neg","exp","log",...}`
* Implementation maps to MLX C API unary ops; output matches input dtype unless op requires float.

**T10. Elementwise binary ops**

* Support: `+ - * / ^` and comparisons `< <= > >= == !=`.
* Signature:

  * `SEXP cpp_mlx_binary(SEXP x, SEXP y, std::string op);`
  * Broadcasting rules: Follow MLX’s broadcasting (like NumPy). Validate shapes; throw R error on incompatible shapes.
* For scalar RHS/LHS, allow numeric and logical scalars (will wrap in 0-d/1-d MLX arrays or handle as scalar op if MLX supports).

**T11. Reductions**

* Support: `sum`, `mean` (overall and along axes), `min`, `max`.

* Signatures:

  * `SEXP cpp_mlx_reduce(SEXP x, std::string op);`            // full reduction -> scalar mlx
  * `SEXP cpp_mlx_reduce_axis(SEXP x, std::string op, int axis, bool keepdims);`

* R wrappers will provide `sum.mlx`, `mean.mlx`, and helper `colMeans.mlx` / `rowMeans.mlx` built on axis reductions.

* Acceptance: Numerically matches R within tolerance for random small arrays.

---

## Milestone 4 — Matrix algebra & helpers

**T12. Transpose, reshape**

* `SEXP cpp_mlx_transpose(SEXP x);`
* `SEXP cpp_mlx_reshape(SEXP x, SEXP new_dim);`

**T13. Matrix multiply**

* `SEXP cpp_mlx_matmul(SEXP a, SEXP b);`

  * Accept shapes: (m×k) %*% (k×n) → (m×n); vectors where meaningful (k)×(k) → scalar.
  * Use MLX matmul op; ensure float32/float64 supported.

**T14. Crossprod & tcrossprod**

* Implement on top of `matmul` + `transpose`, but exploit MLX fused ops if available later.
* R wrappers:

  * `crossprod.mlx(x, y = NULL)` → `t(x) %*% (y %||% x)`
  * `tcrossprod.mlx(x, y = NULL)` → `x %*% t(y %||% x)`

**T15. `t.mlx`, `colMeans.mlx`, `rowMeans.mlx`**

* `t.mlx` → transpose
* `colMeans.mlx` → `mean(x, axis=0)` assuming R column-major (verify axis mapping; likely axis=0 == rows; be explicit and test).
* `rowMeans.mlx` → `mean(x, axis=1)` (ditto).
* Document axis semantics clearly.

---

## Milestone 5 — R S3 class + operator overloading

**T16. Class constructors & converters (R)**

* File: `R/class.R`

  * `new_mlx <- function(ptr, dim, dtype, device) { structure(list(ptr=ptr, dim=dim, dtype=dtype, device=device), class="mlx") }`
  * `as_mlx <- function(x, dtype=c("float32","float64"), device=c("gpu","cpu"))`

    * Coerce `matrix`/`array`/`numeric` into MLX via `cpp_mlx_from_numeric`.
  * `as.matrix.mlx <- function(x, ...) { mlx_eval(x); cpp_mlx_to_numeric(x$ptr) |> structure(dim=x$dim) }`
  * `mlx_eval <- function(x) { cpp_mlx_eval(x$ptr); invisible(x) }`
  * `is.mlx <- function(x) inherits(x, "mlx")`

**T17. Operator overloading**

* File: `R/ops.R`

  * Define S3 method for **`Ops.mlx`** to dispatch elementwise + comparisons.

    ```r
    Ops.mlx <- function(e1, e2 = NULL) {
      op <- .Generic
      if (is.null(e2)) {
        # unary ops: "+" (no-op), "-" (neg)
        if (op == "+") return(e1)
        if (op == "-") return(.mlx_unary(e1, "neg"))
        stop(sprintf("Unary op '%s' not supported for mlx", op))
      }
      # Coerce scalars/matrix to mlx
      if (!is.mlx(e1)) e1 <- as_mlx(e1)
      if (!is.mlx(e2)) e2 <- as_mlx(e2)
      if (op %in% c("+","-","*","/","^")) return(.mlx_binary(e1, e2, op))
      if (op %in% c("==","!=","<","<=",">",">=")) return(.mlx_binary(e1, e2, op))
      stop(sprintf("Op '%s' not supported for mlx", op))
    }
    ```
  * Helper R wrappers call into C++:

    * `.mlx_unary <- function(x, op) new_mlx(cpp_mlx_unary(x$ptr, op), x$dim, x$dtype, x$device)`
    * `.mlx_binary <- function(x, y, op) new_mlx(cpp_mlx_binary(x$ptr, y$ptr, op), broadcast_dim(x,y), promote_dtype(x,y), common_device(x,y))`

* **Matrix multiply `%*%`**

  * In base R `%*%` is a primitive; define an S3 generic shim:

    ```r
    `%*%.mlx` <- function(x, y) {
      if (!is.mlx(x)) x <- as_mlx(x)
      if (!is.mlx(y)) y <- as_mlx(y)
      new_mlx(cpp_mlx_matmul(x$ptr, y$ptr), c(x$dim[1L], y$dim[length(y$dim)]), promote_dtype(x,y), common_device(x,y))
    }
    ```
  * Register in `.onLoad`:

    ```r
    .onLoad <- function(...) {
      # Ensure our method is visible for dispatch
      # (S3 method for primitive is recognized if named `%*%.mlx`)
    }
    ```

**T18. Stats helpers**

* File: `R/stats.R`

  * `sum.mlx`, `mean.mlx` → reductions (full).
  * `colMeans.mlx`, `rowMeans.mlx` → axis reductions + `drop=TRUE` to return 1D `mlx` (dim of length 1 removed) or keepdims + reshape.
  * `t.mlx` → transpose wrapper.
  * `crossprod.mlx`, `tcrossprod.mlx`.

**T19. Class utilities**

* File: `R/utils.R`

  * `print.mlx` → show shape, dtype, device, lazy/evaluated flag; show small preview (e.g., up to 6×6) by evaluating a small slice only (or optionally evaluate fully if size small).
  * `str.mlx` → concise structure.

* Acceptance: Arithmetic, comparisons, `%*%`, col/row means and reductions work between mlx/regular R objects, producing an `mlx` until user calls `as.matrix()` or `mlx_eval()`.

---

## Milestone 6 — Indexing, printing, diagnostics

**T20. Indexing `[ ]`**

* R method:

  ```r
  `[.mlx` <- function(x, i, j, ..., drop = TRUE) {
    # Convert missing to full spans, build slices, call C++ slice
    new_mlx(cpp_mlx_slice(x$ptr, normalize_index(i), normalize_index(j)), new_dim, x$dtype, x$device)
  }
  ```
* C++: `SEXP cpp_mlx_slice(SEXP x, SEXP i_, SEXP j_);`

  * Support integer/real/logical vectors; negative indices throw (or translate).
  * Use MLX slicing ops (start/stop/step per axis).
* Acceptance: `x[ ,1]`, `x[1:5, 3:7]`, logical masks for rows/cols (optional in v1) behave; returns `mlx`.

**T21. Shape/dtype accessors**

* R:

  * `mlx_dim <- function(x) x$dim`
  * `mlx_dtype <- function(x) x$dtype`
  * `dim.mlx <- function(x) x$dim` (S3)
  * `length.mlx <- function(x) prod(x$dim)`

**T22. Error messages**

* Ensure all C++ functions translate exceptions to R errors with actionable messages: incompatible shapes, dtype mismatch, not found MLX op, etc.

---

## Milestone 7 — Device/stream management

**T23. Default device**

* R:

  * `mlx_default_device <- local({ dev <- "gpu"; function(value) { if (!missing(value)) dev <<- match.arg(value, c("gpu","cpu")); dev }})`
  * Used by `as_mlx()` and constructors if device not specified.
* C++:

  * Keep references to MLX CPU/GPU streams (e.g., `MLX_GPU_STREAM`, `MLX_CPU_STREAM` identifiers). Make a small helper to select stream per call.
* Acceptance: Users can switch default to CPU for debugging; all ops route accordingly.

**T24. Synchronization**

* R: `mlx_synchronize(device=c("gpu","cpu"))` → C++ call to stream/device synchronize (if MLX exposes this).
* Acceptance: No outstanding work remains after synchronize.

---

## Milestone 8 — Docs, examples, tests

**T25. Roxygen2 docs**

* Add `#'` docs for all user-facing functions: `as_mlx`, `as.matrix.mlx`, `mlx_eval`, ops overview, `%*%`, `colMeans.mlx`, `rowMeans.mlx`, etc.
* `?mlx` overview man page: explain laziness, evaluation points, device selection, and unified memory concept.

**T26. Vignette**

* `vignettes/getting-started.Rmd`

  * Walkthrough: creating `mlx` arrays, arithmetic, `%*%`, reductions, `colMeans`, evaluation/convert, simple timing demo vs base R for large matmul (note: not a formal benchmark).
  * State Apple-only requirement.

**T27. Tests (testthat)**

* Skip tests if MLX unavailable or device not GPU:

  ```r
  skip_if_not(mlx_available())
  ```
* Tests:

  * Roundtrip `as_mlx` → `as.matrix` equality.
  * Elementwise ops vs base R (small sizes).
  * `%*%` correctness vs base R.
  * `colMeans/rowMeans/sum/mean` equality vs base R.
  * Broadcasting cases.
  * Indexing behavior.
* Tolerances: use `expect_equal(..., tolerance = 1e-6)`.

---

## Milestone 9 — Build, local check, packaging

**T28. Local build**

* Commands:

  * `R CMD build .`
  * `R CMD INSTALL Rmlx_0.0.0.9000.tar.gz`
  * `R CMD check Rmlx_0.0.0.9000.tar.gz --as-cran`
* Accept: no ERRORs; NOTE/WARN only for SystemRequirements acceptable.

**T29. Minimal examples**

* Add examples to function docs that run fast and only touch small arrays to avoid GPU timeouts.

---

## File-by-file stubs (for the agent)

**R/class.R**

```r
#' Create MLX array from R object
#' @export
as_mlx <- function(x, dtype = c("float32","float64"), device = mlx_default_device()) {
  dtype <- match.arg(dtype)
  if (is.mlx(x)) return(x)
  if (is.vector(x)) dim <- length(x) else dim <- dim(x)
  stopifnot(!is.null(dim))
  ptr <- cpp_mlx_from_numeric(as.numeric(x), as.integer(dim), dtype, device)
  structure(list(ptr = ptr, dim = as.integer(dim), dtype = dtype, device = device), class = "mlx")
}

#' Force evaluation
#' @export
mlx_eval <- function(x) { stopifnot(is.mlx(x)); cpp_mlx_eval(x$ptr); invisible(x) }

#' Convert MLX array to base matrix/array
#' @export
as.matrix.mlx <- function(x, ...) {
  mlx_eval(x)
  out <- cpp_mlx_to_numeric(x$ptr)
  dim(out) <- x$dim
  out
}

is.mlx <- function(x) inherits(x, "mlx")
```

**R/ops.R** (skeleton shown earlier)

**R/stats.R**

```r
#' @export
sum.mlx  <- function(x, ...) .mlx_reduce(x, "sum")
#' @export
mean.mlx <- function(x, ...) .mlx_reduce(x, "mean")

#' @export
colMeans.mlx <- function(x, na.rm = FALSE, dims = 1, ...) .mlx_reduce_axis(x, "mean", axis = 0L, keepdims = FALSE)
#' @export
rowMeans.mlx <- function(x, na.rm = FALSE, dims = 1, ...) .mlx_reduce_axis(x, "mean", axis = 1L, keepdims = FALSE)

#' @export
t.mlx <- function(x) new_mlx(cpp_mlx_transpose(x$ptr), rev(x$dim), x$dtype, x$device)

#' @export
crossprod.mlx <- function(x, y = NULL) { if (is.null(y)) y <- x; t(x) %*% y }
#' @export
tcrossprod.mlx <- function(x, y = NULL) { if (is.null(y)) y <- x; x %*% t(y) }
```

**src/Makevars** (generated by `configure`)

```
PKG_CPPFLAGS = -I$(MLX_INCLUDE)
PKG_LIBS     = -L$(MLX_LIB_DIR) -lmlx
```

**src/registration.cpp**

```cpp
#include <Rcpp.h>
using namespace Rcpp;
// forward-declare C++ functions to register with R
// RCPP_MODULE / R_registerRoutines as needed

extern "C" {
  void R_init_Rmlx(DllInfo *dll) {
    R_registerRoutines(dll, NULL, NULL, NULL, NULL);
    R_useDynamicSymbols(dll, TRUE);
  }
}
```

**src/mlx_bindings.cpp** (sketch)

```cpp
#include <Rcpp.h>
#include "mlx_bindings.hpp"
#include <mlx/c_api.h>  // adjust include path per installed MLX

using namespace Rcpp;

// Helpers to unwrap/wrap external pointers, set dims/dtype/device (store those in R side)

SEXP cpp_mlx_from_numeric(SEXP x_, SEXP dim_, SEXP dtype_, SEXP device_) {
  NumericVector x(x_);
  IntegerVector dim(dim_);
  std::string dtype = as<std::string>(dtype_);
  std::string device = as<std::string>(device_);

  // create MLX array with given shape and dtype on device
  // copy x.data() into MLX array
  // return XPtr<MlxArray>(new MlxArray(ptr), true)
}

SEXP cpp_mlx_to_numeric(SEXP xp_) {
  // ensure evaluated
  // copy MLX array data to NumericVector
}

void cpp_mlx_eval(SEXP xp_) {
  // call MLX eval on underlying array/graph
}

// unary, binary, reduce, reduce_axis, transpose, reshape, matmul, slice...
```

---

## Acceptance checklist (Phase 1 complete)

* [ ] Build succeeds on Apple Silicon with MLX installed.
* [ ] `as_mlx()`, `as.matrix.mlx()` roundtrip correct.
* [ ] `+ - * / ^`, comparisons, `%*%` produce correct results vs base R (small sizes).
* [ ] `sum.mlx`, `mean.mlx`, `colMeans.mlx`, `rowMeans.mlx`, `t.mlx`, `crossprod.mlx`, `tcrossprod.mlx` correct.
* [ ] Lazy by default; `as.matrix.mlx()` forces evaluation; `mlx_eval()` works.
* [ ] Indexing `[]` supports common cases.
* [ ] Helpful errors if MLX not found at install, or shape/dtype mismatches at runtime.
* [ ] Vignette explains usage and caveats.

---

## Risks & notes for the agent

* **MLX headers & symbols:** Ensure correct include (e.g., `#include <mlx/c_api.h>`) and link flags; the actual header path & lib name may vary; keep `configure` flexible with env overrides.
* **Dtype:** R is `double` by default; we’ll allow `float32` and `float64`. Decide default (`float32` is faster on GPU; but to match R expectations, maybe default `float64`; document choice).
* **Axis conventions:** Confirm MLX axis numbering vs R’s column-major expectations; lock this down with tests for `colMeans/rowMeans`.
* **Broadcasting:** Implement consistent rules; add tests for scalar + array, vector + matrix.
* **Evaluation semantics:** If MLX needs explicit graph roots or streams for `eval`, store whatever handle is required with each array (or a per-session singleton), and make `cpp_mlx_eval` robust.
* **Thread safety:** Rcpp calls execute on R main thread; ensure MLX usage is safe in that context.
* **Indexing:** Logical indexing may be deferred; start with integer ranges and `:` slices.

---

## Phase 2 (later, not in scope for this handoff)

* Autodiff: wrap MLX grad transforms; expose `mlx_grad(fn, params)` for R closures or provide graph-based differentiation APIs.
* Optimizers: SGD/Adam on `mlx` parameters.
* More linalg: `solve`, `chol`, `svd`, `eigen` (depending on MLX support).
* Datasets/dataloaders, random seeding, fused kernels, compilation.

