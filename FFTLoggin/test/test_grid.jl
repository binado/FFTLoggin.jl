using FFTLoggin
using Test

@testset "grid utilities" begin
    r = 10 .^ range(-2, 2; length = 128)
    @test isapprox(infer_dlog(r), (4 / 127) * log(10); rtol = 1e-12)

    # Non-log-spaced -> error
    @test_throws ArgumentError infer_dlog(collect(1.0:128.0))

    @test FFTLoggin.infer_logc(r; logc = 0.5) == 0.5
    @test FFTLoggin.infer_logc(r; ycenter = 1.0) ≈ 0.0 atol=1e-12
    @test FFTLoggin.infer_logc(r; ymax = 100.0) ≈
          log(100.0 * minimum(r)) rtol=1e-12
    @test FFTLoggin.infer_logc(r; ymin = 0.01) ≈
          log(0.01 * maximum(r)) rtol=1e-12
    @test_throws ArgumentError FFTLoggin.infer_logc(r)

    f = FFTLog(BesselJKernel(0), r; kr = 1.0, lowring = false)
    g = loggrid(f; r = r)
    @test length(g.k) == 128
    @test g.k ≈ exp(0.0) ./ reverse(r) rtol=1e-12

    g2 = loggrid(f; k = g.k)
    @test g2.r ≈ r rtol=1e-12

    fkr = FFTLog(BesselJKernel(0), r; kr = [0.5, 2.0], lowring = false)
    gkr = loggrid(fkr; r = r)
    expected_k = reshape([0.5, 2.0], 1, :) ./ reverse(r; dims = 1)
    @test size(gkr.r) == (128, 2)
    @test size(gkr.k) == (128, 2)
    @test gkr.r ≈ repeat(r, 1, 2) rtol=1e-12
    @test gkr.k ≈ expected_k rtol=1e-12
    @test loggrid(fkr; k = gkr.k).r ≈ gkr.r rtol=1e-12

    rmat = hcat(r, 2 .* r)
    gmat = loggrid(fkr; r = rmat)
    @test size(gmat.k) == size(rmat)
    @test gmat.k ≈ reshape([0.5, 2.0], 1, :) ./ reverse(rmat; dims = 1) rtol=1e-12

    @test_throws DimensionMismatch loggrid(fkr; r = repeat(r, 1, 3))
    @test_throws DimensionMismatch loggrid(f; r = zeros(127))
end
