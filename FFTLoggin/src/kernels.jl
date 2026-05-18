"""
    AbstractKernel

Base type for Mellin transform kernels.

A concrete kernel `K` must implement:

  * `(k::K)(s)` — evaluate the Mellin transform at `s` (scalar or array).
  * `domain(k::K)` — return `(lo, hi)` bounds of the strip of convergence,
    with bounds broadcast-compatible with `real.(s)`.

It may override `isindomain(k, s)`. Default `isindomain` checks that the real
part of `s` lies in `[lo, hi]` and reduces to a scalar `Bool`.
"""
abstract type AbstractKernel end

"""
    domain(k::AbstractKernel)

Return `(lo, hi)`, the strip of convergence of the kernel `k`.
"""
# Declare the generic function so kernel implementations can add methods.
function domain end

"""
    convergence_strip(k::AbstractKernel)

Return `(lo, hi)`, the strip of convergence of the kernel `k`.

This is a descriptive alias for [`domain`](@ref).
"""
convergence_strip(k::AbstractKernel) = domain(k)

"""
    mellin(k::AbstractKernel, s)

Evaluate the Mellin transform of `k` at `s`. Equivalent to `k(s)`.
"""
mellin(k::AbstractKernel, s) = k(s)

"""
    isindomain(k::AbstractKernel, s) -> Bool

Return `true` if all of `real.(s)` lie within the strip of convergence of `k`.
`s` may be a scalar or array-like value.
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

`μ` may be a real scalar or an `AbstractArray`.

Array-valued `μ` is first-class API. It is useful when evaluating many orders
at once, for example line-of-sight integrals vectorized across multipoles
`ℓ`. Kernel evaluation follows Julia broadcasting rules across `μ` and `s`.
"""
struct BesselJKernel{T} <: AbstractKernel
    μ::T
end

domain(k::BesselJKernel) = (-k.μ, _besselj_domain_hi(k.μ))

_besselj_domain_hi(μ::Real) = oftype(float(μ), 1.5)
_besselj_domain_hi(μ::AbstractArray) = fill(oftype(float(zero(eltype(μ))), 1.5), size(μ))

function (k::BesselJKernel)(s)
    return _besselj_call(k.μ, s)
end

@inline _besselj_logvalue(μ, s) = LOG_2 * (s - 1) + loggamma((μ + s) / 2) -
                                  loggamma((μ + 2 - s) / 2)

# Broadcast handles scalar and array combinations of μ and s using Julia rules.
_besselj_call(μ, s) = exp.(_besselj_logvalue.(μ, s))

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
struct SphericalBesselJKernel{T, K <: BesselJKernel} <: AbstractKernel
    ℓ::T
    _inner::K
end

function SphericalBesselJKernel(ℓ)
    return SphericalBesselJKernel(ℓ, BesselJKernel(ℓ .+ 0.5))
end

function domain(k::SphericalBesselJKernel)
    lo, hi = domain(k._inner)
    return (lo .+ 0.5, hi .+ 0.5)
end

function (k::SphericalBesselJKernel)(s)
    SQRT_PI_OVER_2 .* k._inner(s isa AbstractArray ? s .- 0.5 : s - 0.5)
end

# ---------------------------------------------------------------------------
# ShiftedKernel
# ---------------------------------------------------------------------------

"""
    ShiftedKernel(base, ν)

Wrapper representing `s -> base(s + ν)`. Use [`shift`](@ref) to construct.
"""
struct ShiftedKernel{K <: AbstractKernel, T} <: AbstractKernel
    base::K
    ν::T
end

function domain(k::ShiftedKernel)
    lo, hi = domain(k.base)
    return (lo .- k.ν, hi .- k.ν)
end

function isindomain(k::ShiftedKernel, s)
    isindomain(k.base, s isa AbstractArray ? s .+ k.ν : s + k.ν)
end

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
struct DerivativeKernel{K <: AbstractKernel} <: AbstractKernel
    base::K
    order::Int
    function DerivativeKernel(base::K, order::Int) where {K <: AbstractKernel}
        order >= 1 || throw(
            ArgumentError("Expected derivative order to be an integer >= 1, got $order"),
        )
        new{K}(base, order)
    end
end

function domain(k::DerivativeKernel)
    lo, hi = domain(k.base)
    return (lo .+ k.order, hi .+ k.order)
end

function isindomain(k::DerivativeKernel, s)
    isindomain(k.base, s isa AbstractArray ? s .- k.order : s - k.order)
end

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

"""
    shift(k::AbstractKernel, ν)

Return a kernel representing `s -> k(s + ν)`. `ν == 0` returns `k` unchanged.
For `ShiftedKernel`s, the shifts are combined.
"""
function shift(k::AbstractKernel, ν)
    if ν isa Number && ν == 0
        return k
    end
    return ShiftedKernel(k, ν)
end

function shift(k::ShiftedKernel, ν)
    if ν isa Number && ν == 0
        return k
    end
    return ShiftedKernel(k.base, k.ν .+ ν)
end

# ---------------------------------------------------------------------------
# optimal_logcenter
# ---------------------------------------------------------------------------

"""
    optimal_logcenter(kernel, dlog, bias = 0)

Return the optimal log-center parameter (Hamilton 2000, Eq. 30) that minimizes
ringing for the given kernel, scalar log spacing, and scalar bias.

For multiple spacings or biases, use Julia broadcasting:

```julia
optimal_logcenter.(Ref(kernel), dlogs, biases)
```
"""
function optimal_logcenter(kernel::AbstractKernel, dlog::Number, bias::Number = 0.0)
    s = im * pi / dlog + 1 + bias
    arg = angle.(kernel(s))
    return dlog .* arg ./ pi
end
