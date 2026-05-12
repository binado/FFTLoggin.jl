"""
    FFTLog(kernel; n, dlog, bias=0.0, kr=1.0, lowring=true)
    FFTLog(kernel, r::AbstractVector; bias=0.0, kr=1.0, lowring=true)

Pure FFTLog transform. Immutable. Coefficients are precomputed in the
constructor; coordinate management is handled separately via [`loggrid`](@ref).

Sample/transform axis is the **first** axis (column-major, opposite to the
Python package).

For array-valued kernels, the sample/transform axis remains first and kernel
batch axes are trailing.
"""
struct FFTLog{K <: AbstractKernel, T <: AbstractFloat, D, B, R, CT}
    kernel::K
    n::Int
    dlog::D
    bias::B
    kr::R
    coeffs::CT
    _bias_window_forward::Vector{T}
end

"""
    FFTLogWorkspace(fftlog, a; kwargs...)

Shape-specific FFT workspace for repeated transforms with `fftlog`.

The representative input `a` must be valid for `forward(fftlog, a)`. Keyword
arguments are forwarded to FFTW planning. The workspace caches plans for the
representative input shape and for the transform output shape, which may differ
when kernel batch axes broadcast with the input.
"""
struct FFTLogWorkspace{
    T <: AbstractFloat, NI, NO, IRP, ORP, OIP}
    n::Int
    input_shape::NTuple{NI, Int}
    output_shape::NTuple{NO, Int}
    input_complex_shape::NTuple{NI, Int}
    output_complex_shape::NTuple{NO, Int}
    input_rfft_plan::IRP
    output_rfft_plan::ORP
    output_irfft_plan::OIP
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

    T = _real_eltype(coeffs)
    bias_window_forward = _bias_power_law(bias, dlog, Int(n), -1, T)

    return FFTLog{
        typeof(kernel),
        T,
        typeof(dlog),
        typeof(bias),
        typeof(kr_eff),
        typeof(coeffs)
    }(
        kernel,
        Int(n),
        dlog,
        bias,
        kr_eff,
        coeffs,
        bias_window_forward
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

function _sample_shape(f::FFTLog)
    f.coeffs isa AbstractVector ?
    (f.n,) :
    (f.n, size(f.coeffs)[2:end]...)
end

function _broadcast_sample_shape(f::FFTLog, a)
    ndims(a) > 0 || throw(DimensionMismatch("input must have sample axis 1"))
    size(a, 1) == f.n ||
        throw(DimensionMismatch("first axis $(size(a,1)) does not match FFTLog n=$(f.n)"))
    return Broadcast.broadcast_shape(size(a), _sample_shape(f))
end

function _complex_shape(n::Integer, shape::Tuple)
    return (Int(n) ÷ 2 + 1, Base.tail(shape)...)
end

_rfft_axis1(a::AbstractVector) = rfft(a)
_rfft_axis1(a::AbstractArray) = rfft(a, 1)

_irfft_axis1(A::AbstractVector, n::Integer) = irfft(A, Int(n))
_irfft_axis1(A::AbstractArray, n::Integer) = irfft(A, Int(n), 1)

_plan_rfft_axis1(a::AbstractVector; kwargs...) = plan_rfft(a; kwargs...)
_plan_rfft_axis1(a::AbstractArray; kwargs...) = plan_rfft(a, (1,); kwargs...)

function _plan_irfft_axis1(A::AbstractVector, n::Integer; kwargs...)
    plan_irfft(A, Int(n); kwargs...)
end
function _plan_irfft_axis1(A::AbstractArray, n::Integer; kwargs...)
    plan_irfft(A, Int(n), (1,); kwargs...)
end

function FFTLogWorkspace(f::FFTLog, a::AbstractArray{<:Real}; kwargs...)
    input_shape = size(a)
    output_shape = _broadcast_sample_shape(f, a)
    input_complex_shape = _complex_shape(f.n, input_shape)
    output_complex_shape = _complex_shape(f.n, output_shape)

    T = _real_eltype(f.coeffs)
    input_sample = Array{T}(undef, input_shape)
    output_sample = input_shape == output_shape ? input_sample :
                    Array{T}(undef, output_shape)
    output_csample = Array{Complex{T}}(undef, output_complex_shape)

    input_rfft_plan = _plan_rfft_axis1(input_sample; kwargs...)
    output_rfft_plan = input_shape == output_shape ?
                       input_rfft_plan :
                       _plan_rfft_axis1(output_sample; kwargs...)
    output_irfft_plan = _plan_irfft_axis1(output_csample, f.n; kwargs...)

    return FFTLogWorkspace{
        T,
        length(input_shape),
        length(output_shape),
        typeof(input_rfft_plan),
        typeof(output_rfft_plan),
        typeof(output_irfft_plan)
    }(
        f.n,
        input_shape,
        output_shape,
        input_complex_shape,
        output_complex_shape,
        input_rfft_plan,
        output_rfft_plan,
        output_irfft_plan
    )
end

function _rfft(f::FFTLog, a::AbstractArray, ::Nothing)
    return _rfft_axis1(a)
end

function _rfft(f::FFTLog, a::AbstractArray, workspace::FFTLogWorkspace)
    workspace.n == f.n ||
        throw(DimensionMismatch("workspace n=$(workspace.n) does not match FFTLog n=$(f.n)"))
    if size(a) == workspace.input_shape
        return workspace.input_rfft_plan * a
    elseif size(a) == workspace.output_shape
        return workspace.output_rfft_plan * a
    end
    throw(
        DimensionMismatch(
        "workspace real shape mismatch: expected $(workspace.input_shape) or " *
        "$(workspace.output_shape), got $(size(a))",
    ),
    )
end

function _irfft(f::FFTLog, A::AbstractArray, ::Nothing)
    return _irfft_axis1(A, f.n)
end

function _irfft(f::FFTLog, A::AbstractArray, workspace::FFTLogWorkspace)
    workspace.n == f.n ||
        throw(DimensionMismatch("workspace n=$(workspace.n) does not match FFTLog n=$(f.n)"))
    size(A) == workspace.output_complex_shape ||
        throw(
            DimensionMismatch(
            "workspace complex shape mismatch: expected " *
            "$(workspace.output_complex_shape), got $(size(A))",
        ),
        )
    return workspace.output_irfft_plan * A
end

function _multiply_coeffs(A, coeffs)
    out_shape = Broadcast.broadcast_shape(size(A), size(coeffs))
    if out_shape == size(A)
        A .*= coeffs
        return A
    end
    return A .* coeffs
end

function _divide_coeffs(A, coeffs)
    out_shape = Broadcast.broadcast_shape(size(A), size(coeffs))
    if out_shape == size(A)
        A ./= conj.(coeffs)
        return A
    end
    return A ./ conj.(coeffs)
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

_bias_logc(bias::Number, kr::Number, sign::Int) = exp(sign * bias * log(kr))

function _bias_logc(bias, kr::Number, sign::Int)
    return exp.(sign .* bias .* log(kr))
end

function _bias_logc(bias, kr::AbstractArray, sign::Int)
    return exp.(sign .* bias .* log.(kr))
end

# --- Forward / Inverse ------------------------------------------------------

"""
    forward(fftlog, a; workspace=nothing) -> A
    forward(fftlog, a, workspace) -> A

Forward FFTLog transform. `a` may be an `AbstractVector` of length `fftlog.n`
or an array whose first axis has length `fftlog.n`. Trailing axes broadcast
with the kernel batch axes. Pass an `FFTLogWorkspace` created from a compatible
representative input to reuse FFT plans.
"""
function forward(f::FFTLog, a::AbstractArray{<:Real}; workspace::Union{Nothing, FFTLogWorkspace} = nothing)
    return _forward_impl(a, f, workspace)
end

function forward(f::FFTLog, a::AbstractArray{<:Real}, workspace::Union{
        Nothing, FFTLogWorkspace})
    return _forward_impl(a, f, workspace)
end

"""
    inverse(fftlog, A; workspace=nothing) -> a
    inverse(fftlog, A, workspace) -> a

Inverse FFTLog transform. `A` may be an `AbstractVector` of length `fftlog.n`
or an array whose first axis has length `fftlog.n`. Trailing axes broadcast
with the kernel batch axes. Pass an `FFTLogWorkspace` created from a compatible
representative input to reuse FFT plans.
"""
function inverse(f::FFTLog, A::AbstractArray{<:Real}; workspace::Union{Nothing, FFTLogWorkspace} = nothing)
    return _inverse_impl(A, f, workspace)
end

function inverse(f::FFTLog, A::AbstractArray{<:Real}, workspace::Union{
        Nothing, FFTLogWorkspace})
    return _inverse_impl(A, f, workspace)
end

function _forward_impl(a, f::FFTLog, workspace)
    T = _real_eltype(f.coeffs)
    _broadcast_sample_shape(f, a)
    pl = f._bias_window_forward
    blogc = _bias_logc(f.bias, f.kr, -1)

    a_biased = Array{T}(undef, size(a))
    a_biased .= a .* pl
    A = _rfft(f, a_biased, workspace)
    A = _multiply_coeffs(A, f.coeffs)
    out = _irfft(f, A, workspace)
    out_flipped = reverse(out; dims = 1)
    out_flipped .*= pl
    out_flipped .*= blogc
    return out_flipped
end

function _inverse_impl(ak, f::FFTLog, workspace)
    T = _real_eltype(f.coeffs)
    _broadcast_sample_shape(f, ak)
    pl_fwd = f._bias_window_forward
    blogc = _bias_logc(f.bias, f.kr, 1)

    ak_biased = Array{T}(undef, size(ak))
    ak_biased .= ak ./ pl_fwd .* blogc
    A = _rfft(f, ak_biased, workspace)
    A = _divide_coeffs(A, f.coeffs)
    out = _irfft(f, A, workspace)
    out_flipped = reverse(out; dims = 1)
    out_flipped ./= pl_fwd
    return out_flipped
end
