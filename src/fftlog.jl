"""
    FFTLog(kernel; n, dlog, bias=0.0, kr=1.0, lowring=true)
    FFTLog(kernel, r::AbstractVector; bias=0.0, kr=1.0, lowring=true)

Pure FFTLog transform plan. Immutable. Coefficients and FFT plans are
precomputed in the constructor; coordinate management is handled separately
via [`loggrid`](@ref).

Sample/transform axis is the **first** axis (column-major, opposite to the
Python package).

For array-valued kernels, the sample/transform axis remains first and kernel
batch axes are trailing.
"""
struct FFTLog{K <: AbstractKernel, T <: AbstractFloat, C <: Complex, D, B, R, CT, P, IP}
    kernel::K
    n::Int
    dlog::D
    bias::B
    kr::R
    coeffs::CT
    _bias_window_forward::Vector{T}  # forward mask; inverse uses ./ this vector
    fwd_plan::P
    inv_plan::IP
end

# --- Coefficient computation -----------------------------------------------

function _compute_coeffs(kernel::AbstractKernel, n::Integer, kr, dlog, bias)
    ns = n ÷ 2 + 1
    m = collect(0:(ns - 1))
    angle = (2π * im / n) .* m ./ dlog
    s = angle .+ 1 .+ bias
    c = kernel(s)
    logc = log.(kr)
    c = c .* exp.(.-angle .* logc)
    if iseven(n)
        # Realify Nyquist along the first axis.
        if c isa AbstractVector
            c[end] = real(c[end])
        else
            # Batched kernels store coefficients with the Fourier axis first.
            sel = ntuple(i -> i == 1 ? size(c, 1) : Colon(), ndims(c))
            c[sel...] .= real.(c[sel...])
        end
    end
    return c
end

# --- Low-ring snap ----------------------------------------------------------

function _snap_lowring(kernel::AbstractKernel, kr, dlog, bias)
    logc_opt = optimal_logcenter(kernel, dlog, bias)
    logc = log.(kr)
    s = (logc .- logc_opt) ./ dlog
    return exp.(logc_opt .+ round.(s) .* dlog)
end

# --- Domain warning helper -------------------------------------------------

function _warn_if_out_of_domain(kernel::AbstractKernel, bias)
    s = 1 .+ bias
    if !isindomain(kernel, s)
        @warn "FFTLog bias parameter is outside the kernel's strip of convergence" bias
    end
end

# --- Constructors -----------------------------------------------------------

function FFTLog(
        kernel::AbstractKernel;
        n::Integer,
        dlog,
        bias = 0.0,
        kr = 1.0,
        lowring::Bool = true
)
    n > 0 || throw(ArgumentError("n must be positive"))
    _warn_if_out_of_domain(kernel, bias)

    kr_eff = lowring ? _snap_lowring(kernel, kr, dlog, bias) : kr

    coeffs = _compute_coeffs(kernel, Int(n), kr_eff, dlog, bias)

    # Real FFT plans for transforms along axis 1.
    T = _real_eltype(coeffs)
    bias_window_forward = _bias_power_law(bias, dlog, Int(n), -1, T)
    if coeffs isa AbstractVector
        sample = Vector{T}(undef, Int(n))
        fwd_plan = plan_rfft(sample)
        csample = Vector{Complex{T}}(undef, Int(n) ÷ 2 + 1)
        inv_plan = plan_irfft(csample, Int(n))
    else
        sample = Array{T}(undef, Int(n), size(coeffs)[2:end]...)
        fwd_plan = plan_rfft(sample, (1,))
        csample = Array{Complex{T}}(undef, size(coeffs))
        inv_plan = plan_irfft(csample, Int(n), (1,))
    end

    C = Complex{T}
    return FFTLog{
        typeof(kernel),
        T,
        C,
        typeof(dlog),
        typeof(bias),
        typeof(kr_eff),
        typeof(coeffs),
        typeof(fwd_plan),
        typeof(inv_plan)
    }(
        kernel,
        Int(n),
        dlog,
        bias,
        kr_eff,
        coeffs,
        bias_window_forward,
        fwd_plan,
        inv_plan
    )
end

function FFTLog(
        kernel::AbstractKernel,
        r::AbstractVector;
        bias = 0.0,
        kr = 1.0,
        lowring::Bool = true
)
    n = length(r)
    dlog = infer_dlog(r)
    return FFTLog(kernel; n = n, dlog = dlog, bias = bias, kr = kr, lowring = lowring)
end

# Sugar
(f::FFTLog)(a) = forward(f, a)

# --- Helpers ----------------------------------------------------------------

_real_eltype(x::Number) = real(typeof(x)) <: AbstractFloat ? real(typeof(x)) : Float64
_real_eltype(x::AbstractArray) = (T = real(eltype(x)); T <: AbstractFloat ? T : Float64)

_sample_shape(f::FFTLog) = f.coeffs isa AbstractVector ?
                            (f.n,) :
                            (f.n, size(f.coeffs)[2:end]...)

function _broadcast_sample_shape(f::FFTLog, a)
    ndims(a) > 0 || throw(DimensionMismatch("input must have sample axis 1"))
    size(a, 1) == f.n ||
        throw(DimensionMismatch("first axis $(size(a,1)) does not match FFTLog n=$(f.n)"))
    try
        return Broadcast.broadcast_shape(size(a), _sample_shape(f))
    catch err
        if err isa DimensionMismatch
            throw(
                DimensionMismatch(
                    "input shape $(size(a)) does not broadcast with FFTLog sample shape " *
                    "$(_sample_shape(f))"
                )
            )
        end
        rethrow()
    end
end

function _rfft_first_axis(f::FFTLog, a)
    if size(a) == _sample_shape(f)
        return f.fwd_plan * a
    end
    return plan_rfft(a, (1,)) * a
end

function _irfft_first_axis(f::FFTLog, A)
    if size(A) == size(f.coeffs)
        return f.inv_plan * A
    end
    return plan_irfft(A, f.n, (1,)) * A
end

# --- Bias factors ----------------------------------------------------------

function _bias_power_law(
        bias,
        dlog,
        n::Integer,
        sign::Int,
        ::Type{T}
) where {T <: AbstractFloat}
    ic = (n - 1) / 2
    i_minus_ic = collect(T, (0:(n - 1)) .- ic)
    arg = sign .* (i_minus_ic .* dlog) .* bias
    return exp.(arg)
end

function _bias_logc(bias, kr, sign::Int)
    return exp(sign * bias * log(kr))
end

_bias_logc_arr(bias, kr, sign::Int) = exp.(sign .* bias .* log.(kr))

# --- Forward / Inverse ------------------------------------------------------

"""
    forward(fftlog, a) -> A

Forward FFTLog transform. `a` may be an `AbstractVector` of length `fftlog.n`
or an array whose first axis has length `fftlog.n`. Trailing axes broadcast
with the kernel batch axes.
"""
function forward(f::FFTLog, a::AbstractArray{<:Real})
    return _forward_impl(a, f)
end

"""
    inverse(fftlog, A) -> a

Inverse FFTLog transform. `A` may be an `AbstractVector` of length `fftlog.n`
or an array whose first axis has length `fftlog.n`. Trailing axes broadcast
with the kernel batch axes.
"""
function inverse(f::FFTLog, A::AbstractArray{<:Real})
    return _inverse_impl(A, f)
end

function _forward_impl(a, f::FFTLog)
    T = _real_eltype(f.coeffs)
    sample_shape = _broadcast_sample_shape(f, a)
    pl = f._bias_window_forward
    blogc = _bias_logc(f.bias, f.kr, -1)

    a_biased = Array{T}(undef, sample_shape)
    a_biased .= a .* pl
    A = _rfft_first_axis(f, a_biased)
    A .*= f.coeffs
    out = _irfft_first_axis(f, A)
    out_flipped = reverse(out; dims = 1)
    out_flipped .*= pl
    out_flipped .*= blogc
    return out_flipped
end

function _inverse_impl(ak, f::FFTLog)
    T = _real_eltype(f.coeffs)
    sample_shape = _broadcast_sample_shape(f, ak)
    pl_fwd = f._bias_window_forward
    blogc = _bias_logc(f.bias, f.kr, 1)

    ak_biased = Array{T}(undef, sample_shape)
    ak_biased .= ak ./ pl_fwd .* blogc
    A = _rfft_first_axis(f, ak_biased)
    A ./= conj.(f.coeffs)
    out = _irfft_first_axis(f, A)
    out_flipped = reverse(out; dims = 1)
    out_flipped ./= pl_fwd
    return out_flipped
end
