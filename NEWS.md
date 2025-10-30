# Rmlx 0.1.0

* Documented stream-capable arguments across array creation, shape, random,
  and I/O helpers (fixes #8).
* `mlx_default_device()` now forwards to MLX's runtime defaults so device
  switches apply process-wide (fixes #7).
