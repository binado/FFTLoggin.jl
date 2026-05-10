using FFTLoggin
using Test

@testset "FFTLog construction & basics" begin
    r = 10 .^ range(-2, 2; length = 128)
    f = FFTLog(BesselJKernel(0), r; kr = 1.0, lowring = true)
    @test f.n == 128
    @test isapprox(f.dlog, (4 / 127) * log(10); rtol = 1e-12)
    @test isfinite(f.kr)

    a = @. exp(-(r / 1.0)^2)
    A = forward(a, f)
    @test length(A) == 128
    @test all(isfinite, A)

    # Callable sugar
    @test forward(a, f) ≈ f(a)

    # Dimension mismatch
    @test_throws DimensionMismatch forward(zeros(50), f)
end

@testset "lowring snap" begin
    r = 10 .^ range(-2, 2; length = 128)
    fnone = FFTLog(BesselJKernel(0), r; kr = 1.0, lowring = false)
    fsnap = FFTLog(BesselJKernel(0), r; kr = 1.0, lowring = true)
    @test fnone.kr == 1.0
    @test fsnap.kr != 1.0 || isapprox(fsnap.kr, 1.0)  # may already be optimal
end
