# Vendoring MLX - Experimental Branch

This branch explores bundling the MLX library source code with the Rmlx package.

## What Was Done

1. **Bundled MLX source**: Copied MLX library to `src/mlx-src/` (~4.5 MB)
2. **Added copyright attribution**: Created `inst/COPYRIGHTS` with MLX's MIT license
3. **Updated DESCRIPTION**: Added Apple Inc. as copyright holder for bundled MLX
4. **Created configure script**: Builds MLX from source using CMake during package installation
5. **Updated cleanup script**: Removes all generated build artifacts

## Requirements

- macOS on Apple Silicon (M1 or later)
- CMake 3.24 or later
- C++17 compiler
- **Full Xcode** (not just Command Line Tools)

## Key Finding: Metal Compiler Requirement

**The metal shader compiler tool requires full Xcode, not just Command Line Tools.**

During MLX compilation, the build process needs to compile Metal shaders (`.metal` files) using the `metal` and `metallib` tools. These are only available in full Xcode installations, not in the minimal Command Line Tools package.

Error encountered:
```
xcrun: error: unable to find utility "metal", not a developer tool or in PATH
```

## Implications

### For CRAN Submission
- CRAN's build systems may not have full Xcode installed
- Requiring full Xcode (>10 GB download) is a significant barrier for users
- Most R users only have Command Line Tools installed

### Alternative Approaches

1. **Keep current approach** (require system MLX)
   - Users install MLX via `brew install mlx`
   - Package remains small
   - No complex build process
   - ✅ Currently working approach

2. **Pre-compile MLX Metal shaders**
   - Bundle pre-compiled `.metallib` files
   - Still build C++ portions from source
   - Reduces Xcode requirement
   - ⚠️ Would need testing across macOS versions

3. **Provide pre-built binaries** (not CRAN-compatible)
   - Bundle pre-compiled `libmlx.a`
   - Fastest installation
   - ❌ CRAN doesn't accept binary-only packages

## Recommendation

**Do not vendor MLX source code for CRAN submission.**

The current approach (requiring `brew install mlx`) is simpler and more maintainable:
- Smaller package size
- Faster installation
- No Xcode requirement
- Users who need MLX likely already have it or can easily install it
- Standard practice for packages with system dependencies

## Files in This Branch

- `src/mlx-src/`: Bundled MLX source (4.5 MB)
- `configure`: Builds MLX with CMake
- `cleanup`: Removes build artifacts
- `inst/COPYRIGHTS`: MLX license attribution
- `DESCRIPTION`: Updated with Apple Inc. copyright

## Testing

To test this branch (requires full Xcode):
```bash
git checkout experiment-vendor-mlx
./configure
R CMD INSTALL .
```

## Conclusion

While technically possible to vendor MLX, the requirement for full Xcode makes it impractical for general distribution. The experiment demonstrates that:

1. MLX's MIT license permits bundling ✅
2. The build system can be integrated ✅
3. But the runtime requirements are prohibitive ❌

Therefore, maintaining the current approach of requiring system-installed MLX is the better choice.
