using FFTLoggin
using Test
using SpecialFunctions: gamma

@testset "Kernels" begin
    @testset "BesselJKernel basic values" begin
        # M[J0](1) = 1
        k0 = BesselJKernel(0)
        @test isapprox(real(k0(1.0 + 0im)), 1.0; rtol = 1e-12)

        # M[J0](s) = 2^(s-1) Γ(s/2)/Γ(1 - s/2)
        for s in (0.5, 0.9, 1.2)
            expected = 2^(s - 1) * gamma(s / 2) / gamma(1 - s / 2)
            @test isapprox(real(k0(complex(s))), expected; rtol = 1e-10)
        end

        # Domain
        lo, hi = domain(BesselJKernel(2.0))
        @test lo == -2.0
        @test hi ≈ 1.5
        @test convergence_strip(BesselJKernel(2.0)) == domain(BesselJKernel(2.0))

        # Array-valued orders are public API for vectorized kernels.
        μs = [0, 1, 2]
        kout = BesselJKernel(μs)(1.0 + 0im)
        @test size(kout) == size(μs)
        @test all(isfinite, real.(kout))
    end

    @testset "Spherical Bessel kernel" begin
        k = SphericalBesselJKernel(0)
        # j_0(s) Mellin: √(π/2) * M[J_{1/2}](s - 1/2)
        v = real(k(1.0 + 0im))
        # j_0(x) = sin(x)/x, M[sin/x] is well-known
        @test isfinite(v)
    end

    @testset "Derivative kernel" begin
        k = BesselJKernel(0)
        d1 = derive(k, 1)
        d0 = derive(k, 0)
        @test d0 === k
        @test_throws ArgumentError derive(k, -1)
        # Order-1 derivative of J0 at s: -1 * (s-1) * M[J0](s-1)
        s = 1.5 + 0im
        expected = -1 * (s - 1) * k(s - 1)
        @test isapprox(d1(s), expected; rtol = 1e-12)
    end

    @testset "Shift kernel" begin
        k = BesselJKernel(0)
        sk = shift(k, 0.25)
        @test sk(1.0 + 0im) ≈ k(1.25 + 0im) rtol=1e-12
        @test shift(k, 0) === k
        # Combined
        sk2 = shift(sk, 0.25)
        @test sk2 isa FFTLoggin.ShiftedKernel
        @test sk2(1.0 + 0im) ≈ k(1.5 + 0im) rtol=1e-12
    end

    @testset "optimal_logcenter" begin
        v = FFTLoggin.optimal_logcenter(BesselJKernel(0), 0.05, 0.0)
        @test isfinite(v)

        dlogs = [0.04, 0.05]
        biases = [0.0, 0.1]
        vb = FFTLoggin.optimal_logcenter.(Ref(BesselJKernel(0)), dlogs, biases)
        @test size(vb) == size(dlogs)
        @test all(isfinite, vb)
    end
end
