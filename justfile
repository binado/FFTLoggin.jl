# Julia monorepo: workspace root (Project.toml) composes FFTLoggin/ and FFTLogginBenchmark/.

# Resolve dependencies for the active project (default: whole workspace from repo root).
resolve project=".":
    julia --project={{project}} -e 'using Pkg; Pkg.resolve()'

# Run the FFTLoggin test suite.
test:
    julia --project=FFTLoggin -e 'using Pkg; Pkg.test()'

# Generate Fortran reference fixtures used by FFTLoggin tests.
generate-fortran-benchmarks:
    julia --project=. FFTLoggin/test/generate_fortran_benchmarks.jl

# BenchmarkTools driver for scalar vs batched FFTLog forward (instantiate FFTLogginBenchmark first).
benchmark:
    julia --project=FFTLogginBenchmark FFTLogginBenchmark/scripts/benchmark_fftlog.jl

# Build package documentation locally with Documenter.
docs:
    julia --project=docs -e 'using Pkg; Pkg.instantiate()'
    julia --project=docs docs/make.jl

# Format Julia sources under this repo (uses .JuliaFormatter.toml when present).
fmt:
    julia -e 'using JuliaFormatter; format(".")'

repl project="FFTLoggin":
    julia --project={{project}}

sync-notebook:
    jupytext 'notebooks/*.ipynb' --to jl:percent
