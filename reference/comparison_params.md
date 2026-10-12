# Approximate equality parameters

Approximate equality parameters

## Arguments

- a, b:

  MLX arrays or objects coercible to MLX arrays

- rtol:

  Relative tolerance (default: 1e-5)

- atol:

  Absolute tolerance (default: 1e-8)

- equal_nan:

  If `TRUE`, NaN values are considered equal (default: `FALSE`)

## Details

Two values are considered close if:
`abs(a - b) <= (atol + rtol * abs(b))`.

Infinite values with matching signs are considered equal. Supports
NumPy-style broadcasting.
