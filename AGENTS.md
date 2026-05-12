# Repository Guidelines

## Project Structure & Module Organization

This repository is a Julia package named `FFTLoggin`. Public package setup lives in `Project.toml`, with a pinned `Manifest.toml` for reproducible local development. Source code is under `src/`: `FFTLoggin.jl` defines the module, exports the public API, and includes implementation files such as `fftlog.jl`, `grid.jl`, `kernels.jl`, and `utils.jl`. Tests are under `test/`, with `test/runtests.jl` including focused test files by feature. Optional timing benchmarks live under `scripts/` (uses `scripts/Project.toml` with `BenchmarkTools`; see `scripts/benchmark_fftlog.jl`). The `benchmark/` directory is reserved for benchmark assets or scripts.

## Build, Test, and Development Commands

Run commands from the repository root.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Installs the package environment from `Project.toml` and `Manifest.toml`.

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Runs the full test suite in `test/runtests.jl`.

```sh
julia --project=. -e 'using FFTLoggin'
```

Checks that the package loads in the active environment.

```sh
julia --project=scripts scripts/benchmark_fftlog.jl
```

Runs `BenchmarkTools` benchmarks for scalar-kernel vs batched-kernel `forward` (after `Pkg.instantiate()` in the `scripts` environment).

## Coding Style & Naming Conventions

Use idiomatic Julia with 4-space indentation, explicit method signatures, and short internal helper names prefixed with `_` when they are not part of the public API. Public types and kernels use `CamelCase` names, for example `FFTLog`, `BesselJKernel`, and `TupleKernel`; functions use lowercase names such as `forward`, `inverse`, `loggrid`, and `infer_dlog`. Keep the sample/transform axis convention consistent: the first axis is the transform axis. Add docstrings for exported types and functions.

## Testing Guidelines

Tests use Julia's standard `Test` module. Add focused files named `test_<feature>.jl` and include them from `test/runtests.jl`. Prefer small `@testset` blocks with direct numerical tolerances, for example `isapprox(...; rtol = 1e-12)`, and cover vector and first-axis batched matrix behavior where relevant. Run `Pkg.test()` before submitting changes.

## Commit & Pull Request Guidelines

The current history uses concise Conventional Commit style, for example `chore: initial commit` and `refactor: address review comments`. Follow the same pattern: `feat: ...`, `fix: ...`, `test: ...`, `docs: ...`, or `refactor: ...`.

Pull requests should include a short description of the change, the affected API or numerical behavior, and the test command run. Link related issues when available. Include benchmark notes when performance-sensitive FFT, kernel, or allocation behavior changes.

## Agent-Specific Instructions

Keep edits scoped to this Julia package. Do not change generated or unrelated files to satisfy formatting. Preserve compatibility with Julia `1.12` unless the project metadata is intentionally updated (the repo root workspace requires Julia 1.12+).
