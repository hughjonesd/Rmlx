# Subsetting benchmark

Baseline: `c615ba91007bca690e924b22ee29b0fab869dee7` (master before this PR).
Candidate: this PR. Hardware: Apple M2, macOS 27.0.1; R 4.6.0; system MLX 0.32.3.
Both measurements use `devtools::load_all()` with the same native MLX bindings.

`bench_subset_paths.R` creates and evaluates inputs outside timing, warms each
case five times, then records the median of 50 evaluated calls. It includes R
wrapper overhead and `mlx_eval()` synchronization. Assignment starts from an
alias of the evaluated input on every call. Matrices are 1000 x 1000; 3D arrays
are 60 x 70 x 80. See the script for index and replacement sizes.

Copy the script to a temporary location and run it from each revision's checkout:

```sh
Rscript /absolute/path/to/bench_subset_paths.R results.csv
```

The complete results, including cases with little or no improvement, are in
[`bench_subset_paths_results.csv`](bench_subset_paths_results.csv). Results are
single-machine measurements; short operations vary between runs.

| GPU case | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Full matrix read | 0.894 | 0.020 | 44.4x |
| Contiguous rectangle read | 2.377 | 0.119 | 19.9x |
| Contiguous rows read | 1.433 | 0.075 | 19.2x |
| Irregular rectangle read, R indices | 2.583 | 0.745 | 3.5x |
| Logical rectangle read, R indices | 2.730 | 0.544 | 5.0x |
| Contiguous 3D read | 3.568 | 0.069 | 51.5x |
| Contiguous rectangle assignment | 3.240 | 0.710 | 4.6x |
| Scalar rectangle assignment | 3.081 | 0.637 | 4.8x |
| Irregular rectangle assignment, R indices | 3.207 | 0.908 | 3.5x |

On CPU, contiguous rectangle reads improved from 7.744 to 0.087 ms (89.0x),
and contiguous assignments from 14.255 to 0.687 ms (20.8x). Irregular CPU
selections and selections using mlx indices show much smaller improvements.

Sparse coordinate grids contain one vector's worth of positions per axis.
For the irregular rectangle, this reduces the number of explicitly constructed
int32 coordinates from `2 * 600 * 800` to `600 + 800`. No peak GPU-memory claim
is made: the benchmark measures elapsed time, not device memory.
