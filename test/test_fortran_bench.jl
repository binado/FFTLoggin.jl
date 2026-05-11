using FFTLoggin
using Test

const BENCH_DIR = joinpath(@__DIR__, "benchmarks")

function _read_bench(path)
    rows = Vector{Float64}[]
    open(path) do io
        readline(io) # header
        for line in eachline(io)
            isempty(strip(line)) && continue
            push!(rows, parse.(Float64, split(strip(line))))
        end
    end
    return reduce(hcat, rows)'
end

function _parse_bench_filename(name::AbstractString)
    re = r"^benchmark_log10rmin=(?<rmin>-?[\d.]+)_log10rmax=(?<rmax>-?[\d.]+)_n=(?<n>\d+)_mu=(?<mu>\d+)_q=(?<q>-?[\d.]+)_kr=(?<kr>-?[\d.]+)_lowring=(?<lr>[yn])\.txt$"
    m = match(re, name)
    m === nothing && return nothing
    return (
        log10rmin = parse(Float64, m[:rmin]),
        log10rmax = parse(Float64, m[:rmax]),
        n = parse(Int, m[:n]),
        mu = parse(Int, m[:mu]),
        q = parse(Float64, m[:q]),
        kr = parse(Float64, m[:kr]),
        lowring = m[:lr] == "y"
    )
end

f_test(x, mu) = x^(mu + 1) * exp(-x^2 / 2)

@testset "Fortran benchmark agreement" begin
    files = sort(readdir(BENCH_DIR))
    @test length(files) > 0
    rtol = 1e-5
    nchecked = 0
    for fname in files
        params = _parse_bench_filename(fname)
        params === nothing && continue
        data = _read_bench(joinpath(BENCH_DIR, fname))
        k_expected = data[:, 1]
        a_expected = data[:, 2]

        r = 10 .^ range(params.log10rmin, params.log10rmax; length = params.n)
        dlog = (params.log10rmax - params.log10rmin) / (params.n - 1) * log(10)

        # Suppress warnings from out-of-domain bias
        f = redirect_stderr(devnull) do
            FFTLog(
                BesselJKernel(params.mu);
                n = params.n,
                dlog = dlog,
                bias = params.q,
                kr = params.kr,
                lowring = params.lowring
            )
        end

        g = loggrid(f; r = collect(r))
        @test isapprox(g.k, k_expected; rtol = 1e-10)

        fr = f_test.(r, params.mu)
        ak = forward(f, fr)
        @test isapprox(ak, a_expected; rtol = rtol)
        nchecked += 1
    end
    @info "Checked $nchecked Fortran benchmark cases"
end
