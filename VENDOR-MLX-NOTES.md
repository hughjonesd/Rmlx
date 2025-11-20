# Vendoring MLX - Experimental Branch

This branch now vendors MLX on-demand: configure first looks for a
system MLX installation and, if missing, downloads MLX v0.30.0 from
GitHub, supporting both Metal (macOS) and CUDA (Linux) builds alongside
CPU.

## What Was Done

1.  **On-demand MLX source**: Download MLX v0.30.0 into `build/mlx-src/`
    when needed
2.  **Added copyright attribution**: Created `inst/COPYRIGHTS` with
    MLX’s MIT license
3.  **Updated DESCRIPTION**: Credits Apple Inc. for the MLX code
    downloaded at install time
4.  **configure script**: Builds MLX from source via CMake when system
    MLX is absent
    - Always builds CPU backend
    - Prefers Metal backend on Apple Silicon when the `metal` tool is
      available
    - Falls back to CUDA backend on Linux when `nvcc` is present
    - Otherwise builds CPU-only
5.  **Updated cleanup script**: Removes all generated build artifacts

## Requirements

- macOS on Apple Silicon (M1 or later)
- CMake 3.24 or later
- C++17 compiler
- **Full Xcode** (not just Command Line Tools)

## Key Finding: Metal Compiler Requirement

**The metal shader compiler tool requires full Xcode, not just Command
Line Tools.**

During MLX compilation, the build process needs to compile Metal shaders
(`.metal` files) using the `metal` and `metallib` tools. These are only
available in full Xcode installations, not in the minimal Command Line
Tools package.

Error encountered:

    xcrun: error: unable to find utility "metal", not a developer tool or in PATH

## Implications

### For CRAN Submission

- CRAN’s build systems may not have full Xcode installed
- Requiring full Xcode (\>10 GB download) is a significant barrier for
  users
- Most R users only have Command Line Tools installed

### Alternative Approaches

1.  **Keep current approach** (require system MLX)
    - Users install MLX via `brew install mlx`
    - Package remains small
    - No complex build process
    - ✅ Currently working approach
2.  **Pre-compile MLX Metal shaders**
    - Bundle pre-compiled `.metallib` files
    - Still build C++ portions from source
    - Reduces Xcode requirement
    - ⚠️ Would need testing across macOS versions
3.  **Provide pre-built binaries** (not CRAN-compatible)
    - Bundle pre-compiled `libmlx.a`
    - Fastest installation
    - ❌ CRAN doesn’t accept binary-only packages

## Recommendation

**Do not check MLX source into this repo.** Instead, ship a configure
step that prefers system MLX (Homebrew, distro packages, etc.) and
otherwise downloads the exact upstream tarball on demand. This keeps the
package small, avoids forcing all users to install full Xcode, and still
enables Metal/CUDA features when the tooling is present.

## Files in This Branch

- `configure`: Downloads MLX v0.30.0 on demand and builds it with CMake
- `cleanup`: Removes downloaded source/build artifacts (`build/mlx-*`)
- `inst/COPYRIGHTS`: MLX license attribution
- `DESCRIPTION`: Credits Apple Inc. for the MLX sources fetched during
  install

## Testing

``` bash
git checkout experiment-vendor-mlx
./configure   # auto-detects MLX or downloads it
R CMD INSTALL .
```

## Conclusion

Shipping the upstream MLX source in the repository was impractical
(required full Xcode, bloated the tarball, and complicated maintenance).
The current approach auto-detects system MLX and otherwise fetches the
exact upstream release tarball during installation, keeping the package
lightweight while still supporting Metal, CUDA, and CPU builds.
