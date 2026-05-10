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

    @testset "TupleKernel" begin
        k1 = BesselJKernel(0)
        k2 = BesselJKernel(1)
        tk = TupleKernel(k1, k2)
        s = [1.0 + 0im, 1.2 + 0im]
        out = tk(s)
        @test size(out) == (2, 2)
        @test isapprox(out[:, 1], k1(s); rtol = 1e-12)
        @test isapprox(out[:, 2], k2(s); rtol = 1e-12)

        # Flatten nested
        tk2 = TupleKernel(k1, TupleKernel(k2, derive(k1, 1)))
        @test length(tk2.kernels) == 3

        @test_throws ArgumentError TupleKernel()
        @test_throws ArgumentError derive(tk, 1)
        @test_throws ArgumentError shift(tk, 0.1)
    end

    @testset "optimal_logcenter" begin
        v = optimal_logcenter(BesselJKernel(0), 0.05, 0.0)
        @test isfinite(v)
    end
end
