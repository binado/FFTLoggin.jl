using FFTLoggin
using Test

@testset "grid utilities" begin
    r = 10 .^ range(-2, 2; length = 128)
    @test isapprox(infer_dlog(r), (4 / 127) * log(10); rtol = 1e-12)

    # Non-log-spaced -> error
    @test_throws ArgumentError infer_dlog(collect(1.0:128.0))

    @test infer_logc(r; logc = 0.5) == 0.5
    @test infer_logc(r; ycenter = 1.0) ≈ 0.0 atol=1e-12
    @test infer_logc(r; ymax = 100.0) ≈ log(100.0 * minimum(r)) rtol=1e-12
    @test infer_logc(r; ymin = 0.01) ≈ log(0.01 * maximum(r)) rtol=1e-12
    @test_throws ArgumentError infer_logc(r)

    f = FFTLog(BesselJKernel(0), r; kr = 1.0, lowring = false)
    g = loggrid(f; r = r)
    @test length(g.k) == 128
    @test g.k ≈ exp(0.0) ./ reverse(r) rtol=1e-12

    g2 = loggrid(f; k = g.k)
    @test g2.r ≈ r rtol=1e-12
end
