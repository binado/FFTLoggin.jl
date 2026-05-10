# FFTLoggin.jl

Julia port of the [`fftloggin`](https://github.com/binado/fftloggin) Python
package — a vectorized FFTLog implementation for fast Hankel transforms.

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
A = forward(a, fftlog)

g = loggrid(fftlog; r=r)        # NamedTuple (r=..., k=...)
a_back = inverse(A, fftlog)     # roundtrip
```

## Status (Phase 1)

- `BesselJKernel`, `SphericalBesselJKernel`, `ShiftedKernel`,
  `DerivativeKernel`, `TupleKernel`
- `forward`/`inverse` for `Vector` and column-batched `Matrix`
- `loggrid`, `infer_dlog`, `infer_logc`
- Fortran-reference benchmark agreement (216 cases at `rtol = 1e-5`)

See `JULIA_PLAN.md` in the repo root for the full design document and the
Phase 2/3 roadmap.
