# Repository Guidelines

## Project Structure & Module Organization
- `R/` holds exported R wrappers, S3 methods, and roxygen docs; mirror existing files like `ops.R` when adding API surface.
- `src/` contains Rcpp glue to MLX (`mlx_bindings.cpp`, `mlx_ops.cpp`); keep headers in sync with `RcppExports.R`.
- `tests/testthat/` groups unit specs by domain (`test-math.R`, `test-matmul.R`); add new files as `test-feature.R`.
- `vignettes/getting-started.Rmd` introduces workflows; update when adding user-facing features.
- `configure`, `DESCRIPTION`, and `NAMESPACE` manage build-time detection and package metadata; the configure step runs automatically during install.

## Build, Test, and Development Commands
- `R -q -e 'Rcpp::compileAttributes()'` regenerates `RcppExports` after touching headers or `.cpp`.
- `R -q -e 'devtools::document()'` rebuilds NAMESPACE and Rd files from roxygen comments.
- `R -q -e 'devtools::build()'` creates a source tarball; `R -q -e 'devtools::check()'` runs formal package checks.
- `R -q -e 'devtools::test()'` runs the testthat suite; use `R -q -e 'devtools::load_all()'` for rapid iteration.

## Coding Style & Naming Conventions
- Use two-space indents in both R and C++; keep lines under 100 characters to match the current style.
- Prefer snake_case for R helpers (`as_mlx`), and S3 methods as `Generic.class` (`Math.mlx`).
- C++ helpers follow descriptive snake_case and RAII patterns; include `<mlx/mlx.h>` via `mlx_bindings.hpp`.
- Document R functions with roxygen `#'` blocks; let `@export` drive NAMESPACE entries.

## Testing Guidelines
- Write tests with testthat in `tests/testthat`; mirror existing structure and keep scenario-focused blocks within a `test_that`.
- Use CPU-friendly fixtures (small matrices) so GPU and CPU paths run quickly.
- Run `R -q -e 'devtools::test()'` locally; no conditional skips—tests are allowed to fail if MLX is absent.

## Commit & Pull Request Guidelines
- Follow the repository's imperative, capitalized commit style (e.g., `Add rowSums helper`); keep subject lines near 70 characters.
- Each PR should link to issues when relevant, summarize API changes, and note Metal/CPU devices covered.
- Before opening a PR, run `R -q -e 'devtools::document()'`, `R -q -e 'devtools::test()'`, and `R -q -e 'devtools::check()'`; include notable outputs or screenshots for performance-sensitive work.
