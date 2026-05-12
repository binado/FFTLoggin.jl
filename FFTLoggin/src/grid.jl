"""
    infer_dlog(x::AbstractVector; rtol=1e-5) -> Float

Infer the uniform logarithmic spacing of `x = log-spaced grid`. Throws
`ArgumentError` if the array is not (approximately) uniformly log-spaced.
"""
function infer_dlog(x::AbstractVector; rtol::Real = 1e-5)
    n = length(x)
    n >= 2 || throw(ArgumentError("infer_dlog requires length(x) >= 2"))
    logx = log.(x)
    dlog = (logx[end] - logx[1]) / (n - 1)
    diffs = diff(logx)
    if !all(isapprox.(diffs, dlog; rtol = rtol))
        throw(
            ArgumentError(
            "Array is not uniformly log-spaced (expected dlog ≈ $dlog, " *
            "got range [$(minimum(diffs)), $(maximum(diffs))])",
        ),
        )
    end
    return dlog
end

"""
    infer_logc(x; logc=nothing, ycenter=nothing, ymax=nothing, ymin=nothing)

Compute the log-center parameter `log(x_c · y_c)` from `x` and one of `logc`,
`ycenter`, `ymax`, or `ymin`. Mirrors the Python helper.
"""
function infer_logc(
        x::AbstractVector;
        logc = nothing,
        ycenter = nothing,
        ymax = nothing,
        ymin = nothing
)
    xmin, xmax = first(x), last(x)
    xcenter = sqrt(xmin * xmax)

    if logc !== nothing
        return logc
    elseif ycenter !== nothing
        return log(ycenter * xcenter)
    elseif ymax !== nothing
        return log(ymax * xmin)
    elseif ymin !== nothing
        return log(ymin * xmax)
    else
        throw(
            ArgumentError("One of `logc`, `ycenter`, `ymax`, or `ymin` must be provided."),
        )
    end
end

"""
    loggrid(fftlog::FFTLog; r=nothing, k=nothing) -> NamedTuple{(:r,:k)}

Given one log-spaced coordinate array (`r` or `k`), return both arrays related
by `y = exp(logc) ./ reverse(x)` where `logc = log(fftlog.kr)`.
"""
function loggrid(f::FFTLog; r = nothing, k = nothing)
    (r === nothing) ⊻ (k === nothing) ||
        throw(ArgumentError("Provide exactly one of `r` or `k`."))
    logc = log(f.kr isa AbstractArray ? first(f.kr) : f.kr)
    if r !== nothing
        kk = exp(logc) ./ reverse(r)
        return (r = collect(r), k = collect(kk))
    else
        rr = exp(logc) ./ reverse(k)
        return (r = collect(rr), k = collect(k))
    end
end
