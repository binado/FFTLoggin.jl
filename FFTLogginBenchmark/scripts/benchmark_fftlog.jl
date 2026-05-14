# Scalar vs batched-kernel `forward` timing (Fortran-style log10 grid, `f_test` spectrum).
# Run: julia --project=FFTLogginBenchmark FFTLogginBenchmark/scripts/benchmark_fftlog.jl
# Optional argv: comma-separated n, e.g. 64,128,256.

using BenchmarkTools
using BenchmarkTools: prettytime
using FFTLoggin
using Printf

f_test(x, mu) = x^(mu + 1) * exp(-x^2 / 2)

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

        b1 = @benchmark forward!(out1, $f1, $fr, w1) setup=(w1 = FFTLogWorkspace($f1, $fr); out1 = similar($fr, FFTLoggin._broadcast_sample_shape($f1, $fr)))
        b2 = @benchmark forward!(out2, $f2, $fr2, w2) setup=(w2 = FFTLogWorkspace($f2, $fr2); out2 = similar($fr2, FFTLoggin._broadcast_sample_shape($f2, $fr2)))

        t1 = prettytime(time(median(b1)))
        t2 = prettytime(time(median(b2)))
        @printf "%6d  %-18s  %-18s  %10d  %10d\n" n t1 t2 b1.allocs b2.allocs
    end
end

main()
