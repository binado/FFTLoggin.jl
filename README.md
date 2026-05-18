# FFTLoggin.jl

Julia port of the [`fftloggin`](https://github.com/binado/fftloggin) Python
package — a vectorized FFTLog implementation for fast Hankel transforms.

[Documentation](https://binado.github.io/FFTLoggin.jl/dev/)

> **Sample-axis convention.** The transform/sample axis is the **first** axis
> (Julia is column-major). This is the opposite of the Python package, which
> uses the last axis.

## Installation

This package lives under `julia/` in the parent `fftloggin` repository (Phase 1).
From the repository root:

```julia
using Pkg
Pkg.activate("julia")
Pkg.instantiate()
```

## Quickstart

```julia
using FFTLoggin

r = 10 .^ range(-2, 2; length=128)
fftlog = FFTLog(BesselJKernel(0), r; kr=1.0)

a = @. exp(-(r/1.0)^2)
A = forward(fftlog, a)

g = loggrid(fftlog; r=r)        # NamedTuple (r=..., k=...)
a_back = inverse(fftlog, A)     # roundtrip
```

## Shapes and batching

Axis 1 is reserved for FFTLog samples. Every remaining axis participates in
Julia-style broadcasting across signal batches, array-valued kernel parameters,
and array-valued grid centers (`kr`).

```julia
n = 128
r = 10 .^ range(-2, 2; length=n)
a = @. exp(-r^2)

# Vector input, scalar kernel: output is a vector.
f0 = FFTLog(BesselJKernel(0), r)
size(forward(f0, a)) == (n,)

# Matrix input: columns are independent signal batches.
signals = hcat(a, 2a, 0.5a)
size(forward(f0, signals)) == (n, 3)

# Array-valued kernel parameters add trailing output axes.
orders = reshape([0.0, 1.0, 2.0], 1, :)
forders = FFTLog(BesselJKernel(orders); n=n, dlog=infer_dlog(r), lowring=false)
size(forward(forders, a)) == (n, 3)

# Grid centers are batched the same way; axis 1 is the r <-> k correspondence.
fkr = FFTLog(BesselJKernel(0); n=n, dlog=infer_dlog(r), kr=[0.8, 1.0, 1.2], lowring=false)
size(loggrid(fkr; r=r).k) == (n, 3)
```

Inputs whose trailing axes are not mutually broadcast-compatible throw
`DimensionMismatch`.

## Workspaces

Use `FFTLogWorkspace` when repeatedly transforming arrays with the same
`FFTLog` and shape pattern.

```julia
ws = FFTLogWorkspace(f0, signals)
out = similar(signals)

forward!(out, f0, signals, ws)
inverse!(out, f0, out, ws)
```

A workspace caches FFT plans for a representative input shape and the
corresponding broadcast output shape. Later inputs must match one of those
cached shapes, and mutating outputs must match the workspace output shape.

## Advanced utilities

Advanced helper functions are available with qualified names, for example
`FFTLoggin.infer_logc`, `FFTLoggin.optimal_logcenter`,
`FFTLoggin.isindomain`, and `FFTLoggin.mellin`.

## Status (Phase 1)

- `BesselJKernel`, `SphericalBesselJKernel`, `ShiftedKernel`,
  `DerivativeKernel`
- `forward`/`inverse` for `Vector` and column-batched `Matrix`
- `loggrid`, `infer_dlog`
- Fortran-reference benchmark agreement (216 cases at `rtol = 1e-5`)

See `JULIA_PLAN.md` in the repo root for the full design document and the
Phase 2/3 roadmap.
