"""
    AbstractKernel

Base type for Mellin transform kernels.

A concrete kernel `K` must implement:

  * `(k::K)(s)` — evaluate the Mellin transform at `s` (scalar or array).
  * `domain(k::K)` — return `(lo, hi)` bounds of the strip of convergence.

It may override `isindomain(k, s)`. Default `isindomain` checks that the real
part of `s` lies in `[lo, hi]` and reduces to a scalar `Bool`.
"""
abstract type AbstractKernel end

"""
    domain(k::AbstractKernel)

Return `(lo, hi)`, the strip of convergence of the kernel `k`.
"""
function domain end

"""
    mellin(k::AbstractKernel, s)

Evaluate the Mellin transform of `k` at `s`. Equivalent to `k(s)`.
"""
mellin(k::AbstractKernel, s) = k(s)

"""
    isindomain(k::AbstractKernel, s) -> Bool

Return `true` if all of `real.(s)` lie within the strip of convergence of `k`.
"""
function isindomain(k::AbstractKernel, s)
    lo, hi = domain(k)
    sr = real.(s)
    return all((sr .>= lo) .& (sr .<= hi))
end

# ---------------------------------------------------------------------------
# BesselJKernel
# ---------------------------------------------------------------------------

"""
    BesselJKernel(μ)

Mellin transform kernel of the Bessel function ``J_μ``.

```math
M[J_μ](s) = 2^{s-1}\\,\\frac{Γ((μ+s)/2)}{Γ((μ+2-s)/2)}
```

`μ` may be a real scalar or an `AbstractArray`. Integer values are promoted to
`Float64`.
"""
struct BesselJKernel{T} <: AbstractKernel
    μ::T
end

BesselJKernel(μ::Integer) = BesselJKernel(float(μ))
BesselJKernel(μ::AbstractArray{<:Integer}) = BesselJKernel(float.(μ))

domain(k::BesselJKernel) = (-k.μ, oftype(_one_like(k.μ), 1.5) .* _ones_like(k.μ))

_one_like(x::Real) = float(x)
_one_like(x::AbstractArray) = float(zero(eltype(x)))
_ones_like(x::Real) = one(float(x))
_ones_like(x::AbstractArray) = ones(float(eltype(x)), size(x))

function (k::BesselJKernel)(s)
    return _besselj_call(k.μ, s)
end

@inline _besselj_logvalue(μ, s) =
    LOG_2 * (s - 1) + loggamma((μ + s) / 2) - loggamma((μ + 2 - s) / 2)

# Scalar μ, scalar or array s
_besselj_call(μ::Real, s::Number) = exp(_besselj_logvalue(μ, s))
_besselj_call(μ::Real, s::AbstractArray) = exp.(_besselj_logvalue.(μ, s))

# Array μ: broadcasting against s using Julia rules. The user is responsible
# for shaping μ and s so they broadcast correctly.
_besselj_call(μ::AbstractArray, s) = exp.(_besselj_logvalue.(μ, s))

# ---------------------------------------------------------------------------
# SphericalBesselJKernel — standalone, dispatches on its own type
# ---------------------------------------------------------------------------

"""
    SphericalBesselJKernel(ℓ)

Mellin transform of the spherical Bessel function `j_ℓ`. Implemented as
`√(π/2) · BesselJKernel(ℓ + 1/2)(s - 1/2)` internally; defined as its own
struct so that downstream operations (`derive`, `shift`, etc.) dispatch on the
spherical type when needed.
"""
struct SphericalBesselJKernel{T} <: AbstractKernel
    ℓ::T
    _inner::BesselJKernel{T}
end

function SphericalBesselJKernel(ℓ)
    ℓf = ℓ isa Integer ? float(ℓ) : ℓ
    if ℓf isa AbstractArray
        ℓf = float.(ℓf)
        return SphericalBesselJKernel(ℓf, BesselJKernel(ℓf .+ 0.5))
    else
        return SphericalBesselJKernel(ℓf, BesselJKernel(ℓf + 0.5))
    end
end

function domain(k::SphericalBesselJKernel)
    lo, hi = domain(k._inner)
    return (lo .+ 0.5, hi .+ 0.5)
end

(k::SphericalBesselJKernel)(s) = SQRT_PI_OVER_2 .* k._inner(s isa AbstractArray ? s .- 0.5 : s - 0.5)

# ---------------------------------------------------------------------------
# ShiftedKernel
# ---------------------------------------------------------------------------

"""
    ShiftedKernel(base, ν)

Wrapper representing `s -> base(s + ν)`. Use [`shift`](@ref) to construct.
"""
struct ShiftedKernel{K<:AbstractKernel, T} <: AbstractKernel
    base::K
    ν::T
end

function domain(k::ShiftedKernel)
    lo, hi = domain(k.base)
    return (lo .- k.ν, hi .- k.ν)
end

isindomain(k::ShiftedKernel, s) = isindomain(k.base, s isa AbstractArray ? s .+ k.ν : s + k.ν)

(k::ShiftedKernel)(s) = k.base(s isa AbstractArray ? s .+ k.ν : s + k.ν)

# ---------------------------------------------------------------------------
# DerivativeKernel
# ---------------------------------------------------------------------------

"""
    DerivativeKernel(base, order::Int)

Kernel representing the `order`-th derivative of `base` via the Mellin
property
```math
M[d^n f / dr^n](s) = (-1)^n\\,\\frac{Γ(s)}{Γ(s-n)}\\,M[f](s-n).
```
"""
struct DerivativeKernel{K<:AbstractKernel} <: AbstractKernel
    base::K
    order::Int
    function DerivativeKernel(base::K, order::Int) where {K<:AbstractKernel}
        order >= 1 || throw(ArgumentError(
            "Expected derivative order to be an integer >= 1, got $order"))
        new{K}(base, order)
    end
end

function domain(k::DerivativeKernel)
    lo, hi = domain(k.base)
    return (lo .+ k.order, hi .+ k.order)
end

isindomain(k::DerivativeKernel, s) = isindomain(k.base, s isa AbstractArray ? s .- k.order : s - k.order)

function (k::DerivativeKernel)(s)
    n = k.order
    sign = iseven(n) ? 1 : -1
    factor = _falling_factorial(s, n)
    return sign .* factor .* k.base(s isa AbstractArray ? s .- n : s - n)
end

@inline function _falling_factorial(s::Number, n::Int)
    p = one(s)
    @inbounds for j in 1:n
        p *= (s - j)
    end
    return p
end
@inline function _falling_factorial(s::AbstractArray, n::Int)
    return _falling_factorial.(s, n)
end

# ---------------------------------------------------------------------------
# TupleKernel
# ---------------------------------------------------------------------------

"""
    TupleKernel(kernels...)

Type-stable composition of kernels. Forward/inverse with a vector input
produces a `Matrix` whose columns are the per-kernel results. Empty tuples are
disallowed; nested `TupleKernel`s are flattened at construction.
"""
struct TupleKernel{Ks<:Tuple{Vararg{AbstractKernel}}} <: AbstractKernel
    kernels::Ks
end

function TupleKernel(ks::AbstractKernel...)
    isempty(ks) && throw(ArgumentError("TupleKernel requires at least one kernel"))
    flat = _flatten_tuple_kernels(ks)
    return TupleKernel{typeof(flat)}(flat)
end

@inline _flatten_tk(acc::Tuple) = acc
@inline _flatten_tk(acc::Tuple, k::AbstractKernel, rest...) =
    _flatten_tk((acc..., k), rest...)
@inline _flatten_tk(acc::Tuple, k::TupleKernel, rest...) =
    _flatten_tk((acc..., k.kernels...), rest...)
@inline _flatten_tuple_kernels(ks::Tuple) = _flatten_tk((), ks...)

function domain(k::TupleKernel)
    los = map(x -> domain(x)[1], k.kernels)
    his = map(x -> domain(x)[2], k.kernels)
    return los, his
end

function isindomain(k::TupleKernel, s)
    return all(isindomain(kk, s) for kk in k.kernels)
end

function (k::TupleKernel)(s)
    parts = map(kk -> kk(s), k.kernels)
    # Stack along a new last axis so the per-kernel batch dimension is trailing.
    return _stack_kernel_outputs(parts)
end

_stack_kernel_outputs(parts::Tuple{Vararg{Number}}) = collect(parts)
function _stack_kernel_outputs(parts::Tuple)
    # All elements either scalars or arrays; broadcast to common shape and stack.
    bs = Broadcast.broadcast_shape(map(size, parts)...)
    arrs = map(p -> p isa AbstractArray ? (size(p) == bs ? p : broadcast(identity, p, ones(eltype(p), bs))) : fill(p, bs), parts)
    return cat(arrs...; dims = ndims(arrs[1]) + 1)
end

# ---------------------------------------------------------------------------
# derive / shift helpers
# ---------------------------------------------------------------------------

"""
    derive(k::AbstractKernel, order::Integer = 1)

Return a new kernel representing the `order`-th derivative of `k`. `order = 0`
returns `k` unchanged. Negative orders raise `ArgumentError`.
"""
function derive(k::AbstractKernel, order::Integer = 1)
    order < 0 && throw(ArgumentError("derive order must be >= 0, got $order"))
    order == 0 && return k
    return DerivativeKernel(k, Int(order))
end

derive(::TupleKernel, order::Integer = 1) =
    throw(ArgumentError("derive on TupleKernel is not defined; construct " *
                        "TupleKernel(derive.(kernels, order)...) explicitly"))

"""
    shift(k::AbstractKernel, ν)

Return a kernel representing `s -> k(s + ν)`. `ν == 0` returns `k` unchanged.
For `ShiftedKernel`s, the shifts are combined.
"""
function shift(k::AbstractKernel, ν)
    νf = ν isa Integer ? float(ν) : ν
    if νf isa Number && νf == 0
        return k
    end
    return ShiftedKernel(k, νf)
end

function shift(k::ShiftedKernel, ν)
    νf = ν isa Integer ? float(ν) : ν
    if νf isa Number && νf == 0
        return k
    end
    return ShiftedKernel(k.base, k.ν .+ νf)
end

shift(::TupleKernel, ν) =
    throw(ArgumentError("shift on TupleKernel is not defined; construct " *
                        "TupleKernel(shift.(kernels, ν)...) explicitly"))

# ---------------------------------------------------------------------------
# optimal_logcenter
# ---------------------------------------------------------------------------

"""
    optimal_logcenter(kernel, dlog, bias = 0)

Return the optimal log-center parameter (Hamilton 2000, Eq. 30) that minimizes
ringing for the given kernel, log spacing, and bias.
"""
function optimal_logcenter(kernel::AbstractKernel, dlog, bias = 0.0)
    s = im * pi ./ dlog .+ 1 .+ bias
    arg = angle.(kernel(s))
    return dlog .* arg ./ pi
end
