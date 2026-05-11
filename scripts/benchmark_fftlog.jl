# Benchmark FFTLog `forward` for two kernel configurations used in the Fortran
# reference test (`test/test_fortran_bench.jl`):
#
#   1) Vector sample `fr` on a log-spaced grid, scalar `BesselJKernel(μ)`,
#      scalar `dlog`, scalar `bias`.
#   2) Same vector `fr`, `BesselJKernel` with batch orders `μ` (row-shaped for
#      broadcasting), same scalar `dlog` and `bias`.
#
# Run from the repository root (`scripts/Project.toml` provides BenchmarkTools;
# the package itself is loaded via `LOAD_PATH`, see below):
#
#   julia --project=scripts scripts/benchmark_fftlog.jl
#
# Optional first argument: comma-separated grid sizes, e.g. `64,128,256`.
#
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, REPO_ROOT)

using BenchmarkTools
using BenchmarkTools: prettytime
using FFTLoggin
using Printf

# Same test spectrum as in `test/test_fortran_bench.jl`.
f_test(x, mu) = x^(mu + 1) * exp(-x^2 / 2)

"""
Build `(r, dlog, fr)` matching the Fortran harness: uniform `log10(r)` grid and
scalar `dlog`, `bias`.
"""
function _fortran_style_arrays(n::Integer, log10rmin, log10rmax, mu_fr::Real)
    r = 10 .^ range(log10rmin, log10rmax; length = n)
    dlog = (log10rmax - log10rmin) / (n - 1) * log(10)
    fr = f_test.(r, mu_fr)
    return r, dlog, fr
end

function _build_fftlog_scalar(
        n::Integer,
        log10rmin,
        log10rmax,
        mu::Real,
        q,
        kr,
        lowring::Bool,
        mu_fr::Real,
)
    _, dlog, fr = _fortran_style_arrays(n, log10rmin, log10rmax, mu_fr)
    f = redirect_stderr(devnull) do
        FFTLog(
            BesselJKernel(mu);
            n = n,
            dlog = dlog,
            bias = q,
            kr = kr,
            lowring = lowring,
        )
    end
    return f, fr
end

function _build_fftlog_array_kernel(
        n::Integer,
        log10rmin,
        log10rmax,
        mu_row::AbstractMatrix{<:Real},
        q,
        kr,
        lowring::Bool,
        mu_fr::Real,
)
    _, dlog, fr = _fortran_style_arrays(n, log10rmin, log10rmax, mu_fr)
    f = redirect_stderr(devnull) do
        FFTLog(
            BesselJKernel(mu_row);
            n = n,
            dlog = dlog,
            bias = q,
            kr = kr,
            lowring = lowring,
        )
    end
    return f, fr
end

function _parse_ns(arg::Union{Nothing,String})
    arg === nothing && return Int[64, 128, 256, 512, 1024]
    parts = split(arg, ','; keepempty = false)
    isempty(parts) && return Int[64, 128, 256, 512, 1024]
    return parse.(Int, strip.(parts))
end

function main()
    ns = _parse_ns(get(ARGS, 1, nothing))

    # Representative Fortran-style parameters (scalar `dlog`, `bias`, `kr`).
    log10rmin = -4.0
    log10rmax = 4.0
    mu_scalar = 0
    mu_row = reshape([0.0, 1.0, 2.0], 1, :)
    q = 0.0
    kr = 1.0
    lowring = false
    mu_fr = 0.0

    println("FFTLog forward benchmark (Fortran-style grid + f_test spectrum)")
    println("  log10(r) ∈ [$log10rmin, $log10rmax], bias=$q, kr=$kr, lowring=$lowring")
    println("  Case 1: BesselJKernel($mu_scalar)")
    println("  Case 2: BesselJKernel with orders ", vec(mu_row), " (batch axis)")
    println()

    @printf "%6s  %-18s  %-18s  %10s  %10s\n" "n" "scalar (median)" "array (median)" "scalar allocs" "array allocs"
    println(repeat("-", 92))

    for n in ns
        f1, fr = _build_fftlog_scalar(
            n,
            log10rmin,
            log10rmax,
            mu_scalar,
            q,
            kr,
            lowring,
            mu_fr,
        )
        f2, fr2 = _build_fftlog_array_kernel(
            n,
            log10rmin,
            log10rmax,
            mu_row,
            q,
            kr,
            lowring,
            mu_fr,
        )
        @assert fr ≈ fr2

        b1 = @benchmark forward($f1, $fr)
        b2 = @benchmark forward($f2, $fr2)

        t1 = prettytime(time(median(b1)))
        t2 = prettytime(time(median(b2)))
        @printf "%6d  %-18s  %-18s  %10d  %10d\n" n t1 t2 b1.allocs b2.allocs
    end

    println()
    println("Tip: pass grid sizes as first argument, e.g. `192,256`.")
end

main()
