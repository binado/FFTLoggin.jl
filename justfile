# Julia monorepo: workspace root (Project.toml) composes FFTLoggin/ and FFTLogginBenchmark/.

# Resolve dependencies for the active project (default: whole workspace from repo root).
resolve project=".":
    julia --project={{project}} -e 'using Pkg; Pkg.resolve()'

# Run the FFTLoggin test suite.
test:
    julia --project=FFTLoggin -e 'using Pkg; Pkg.test()'

# BenchmarkTools driver for scalar vs batched FFTLog forward (instantiate FFTLogginBenchmark first).
benchmark:
    julia --project=FFTLogginBenchmark FFTLogginBenchmark/scripts/benchmark_fftlog.jl

# Format Julia sources under this repo (uses .JuliaFormatter.toml when present).
fmt:
    julia -e 'using JuliaFormatter; format(".")'

repl project="FFTLoggin":
    julia --project={{project}}

sync-notebook:
    jupytext 'notebooks/*.ipynb' --to jl:percent
