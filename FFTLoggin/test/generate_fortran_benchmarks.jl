#!/usr/bin/env julia

using Downloads
using Printf
using SHA

const TEST_DIR = @__DIR__
const BUILD_DIR = joinpath(TEST_DIR, "benchmark")
const OUTPUT_DIR = joinpath(TEST_DIR, "benchmarks")
const EXECUTABLE = joinpath(BUILD_DIR, "fftlogtest")

const FFTLOG_URL = "https://jila.colorado.edu/~ajsh/FFTLog/fftlog.tgz"
const FFTLOG_SHA256 = "5548708c1c80c8d00ab87036c8c8fc419a40b981def0699ebb9f26d21ecac2e7"

const REQUIRED_FILES = (
    "fftlog.f",
    "fftlogtest.f",
    "cdgamma.f",
    "drfftb.f",
    "drfftf.f",
    "drffti.f"
)

const FFTLOGTEST_PATCHES = (
    (
        ("dlogr=(logrmax-logrmin)/n",),
        "dlogr=(logrmax-logrmin)/(n-1)"
    ),
    (
        (
            "write (unit,'(3es25)') k,a(i),k**(mu+1.d0)*exp(-k**2/2.d0)",
            "write (unit,'(3g24.16)') k,a(i),k**(mu+1.d0)*exp(-k**2/2.d0)"
        ),
        "write (unit,'(3es30.16e3)') k,a(i),k**(mu+1.d0)*exp(-k**2/2.d0)"
    )
)

const LOG10R_PAIRS = ((-1, 1), (-2, 2))
const N_VALUES = (63, 64)
const MU_VALUES = (0, 1, 2)
const Q_VALUES = (-0.5, 0, 0.5)
const KR_VALUES = (0.1, 1, 10)
const LOWRING_VALUES = (true, false)

function _expected_filenames()
    filenames = String[]
    for (log10rmin, log10rmax) in LOG10R_PAIRS,
        n in N_VALUES,
        mu in MU_VALUES,
        q in Q_VALUES,
        kr in KR_VALUES,
        lowring in LOWRING_VALUES
        lowring_str = lowring ? "y" : "n"
        push!(
            filenames,
            "benchmark_log10rmin=$(log10rmin)_log10rmax=$(log10rmax)_n=$(n)_mu=$(mu)_q=$(q)_kr=$(kr)_lowring=$(lowring_str).txt"
        )
    end
    return filenames
end

function _benchmarks_complete()
    filenames = _expected_filenames()
    return all(filename -> isfile(joinpath(OUTPUT_DIR, filename)), filenames)
end

function _sha256_file(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(sha256(io))
    end
end

function _copy_required_sources(extract_dir::AbstractString)
    source_by_name = Dict{String, String}()
    for (root, _, files) in walkdir(extract_dir)
        for file in files
            if file in REQUIRED_FILES
                source_by_name[file] = joinpath(root, file)
            end
        end
    end

    missing = setdiff(collect(REQUIRED_FILES), collect(keys(source_by_name)))
    isempty(missing) ||
        error("missing Fortran source files after extraction: $(join(missing, ", "))")

    mkpath(BUILD_DIR)
    for file in REQUIRED_FILES
        cp(source_by_name[file], joinpath(BUILD_DIR, file); force = true)
    end
    return nothing
end

function _download_fortran_source()
    mkpath(BUILD_DIR)
    mktempdir() do tmpdir
        archive = joinpath(tmpdir, "fftlog.tgz")
        extract_dir = joinpath(tmpdir, "fftlog")
        mkpath(extract_dir)

        println("Downloading FFTLog source from $FFTLOG_URL")
        Downloads.download(FFTLOG_URL, archive)

        digest = _sha256_file(archive)
        digest == FFTLOG_SHA256 ||
            error("FFTLog source checksum mismatch: expected $FFTLOG_SHA256, got $digest")

        run(`tar -xzf $archive -C $extract_dir`)
        _copy_required_sources(extract_dir)
    end
    return nothing
end

function _apply_patches()
    path = joinpath(BUILD_DIR, "fftlogtest.f")
    isfile(path) || error("fftlogtest.f not found in $BUILD_DIR")

    content = read(path, String)
    for (olds, new) in FFTLOGTEST_PATCHES
        old_index = findfirst(old -> contains(content, old), olds)
        if old_index !== nothing
            old = olds[old_index]
            content = replace(content, old => new)
            println("Applied patch: $(_preview(old))...")
        elseif contains(content, new)
            println("Patch already applied: $(_preview(new))...")
        else
            @warn "patch pattern not found" alternatives = olds
        end
    end
    write(path, content)
    return nothing
end

function _compiler()
    fc = get(ENV, "FC", "gfortran")
    try
        run(pipeline(Cmd([fc, "--version"]); stdout = devnull, stderr = devnull))
    catch err
        error("Fortran compiler `$fc` is not available. Set FC or install gfortran.", err)
    end
    return fc
end

_preview(s::AbstractString) = first(s, min(40, length(s)))

function _build_executable()
    fc = _compiler()
    source_files = [joinpath(BUILD_DIR, file) for file in REQUIRED_FILES]
    cmd = Cmd(vcat([fc, "--std=legacy", "-O2", "-o", EXECUTABLE], source_files))

    println("Building fftlogtest with $fc")
    run(cmd)
    chmod(EXECUTABLE, 0o755)
    return nothing
end

function _ensure_executable()
    isfile(EXECUTABLE) && return nothing

    sources_exist = all(file -> isfile(joinpath(BUILD_DIR, file)), REQUIRED_FILES)
    if !sources_exist
        _download_fortran_source()
        _apply_patches()
    end

    _build_executable()
    return nothing
end

function _run_benchmark(log10rmin, log10rmax, n, mu, q, kr, lowring, filename)
    lowring_input = lowring ? "y" : "n"
    input = "$log10rmin $log10rmax\n$n\n$mu\n$q\n$kr\n$lowring_input\n$filename\n"
    cmd = Cmd(Cmd([EXECUTABLE]); dir = OUTPUT_DIR)
    run(pipeline(cmd; stdin = IOBuffer(input), stdout = devnull))
    return nothing
end

function _normalize_fortran_output(path::AbstractString)
    isfile(path) || return false
    content = read(path, String)
    updated = replace(content, r"(?<![EeDd])([+-]?\d*\.\d+)([+-]\d{2,3})" => s"\1e\2")
    if updated != content
        write(path, updated)
    end
    return true
end

function main()
    if _benchmarks_complete()
        println("Fortran benchmark fixtures already exist: $(length(_expected_filenames())) files")
        return nothing
    end

    _ensure_executable()
    mkpath(OUTPUT_DIR)

    filenames = _expected_filenames()
    errors = Pair{String, String}[]
    count = 0
    total = length(filenames)

    println("Generating $total Fortran benchmark files")
    for (log10rmin, log10rmax) in LOG10R_PAIRS,
        n in N_VALUES,
        mu in MU_VALUES,
        q in Q_VALUES,
        kr in KR_VALUES,
        lowring in LOWRING_VALUES
        count += 1
        lowring_str = lowring ? "y" : "n"
        filename = "benchmark_log10rmin=$(log10rmin)_log10rmax=$(log10rmax)_n=$(n)_mu=$(mu)_q=$(q)_kr=$(kr)_lowring=$(lowring_str).txt"
        path = joinpath(OUTPUT_DIR, filename)

        @printf("[%3d/%3d] Running: %s... ", count, total, filename)
        try
            _run_benchmark(log10rmin, log10rmax, n, mu, q, kr, lowring, filename)
            if _normalize_fortran_output(path)
                println("ok")
            else
                println("missing")
                push!(errors, filename => "file not created")
            end
        catch err
            println("failed")
            push!(errors, filename => sprint(showerror, err))
        end
    end

    if !isempty(errors)
        for (filename, err) in errors[begin:min(end, 10)]
            println("ERROR: $filename: $err")
        end
        length(errors) <= 10 || println("... and $(length(errors) - 10) more errors")
        error("failed to generate $(length(errors)) Fortran benchmark files")
    end

    _benchmarks_complete() || error("Fortran benchmark fixture set is incomplete")
    println("Generated $(length(filenames)) Fortran benchmark files")
    return nothing
end

main()
