using FFTLoggin
using Test

@testset "Batched signals (Matrix columns)" begin
    n = 128
    r = 10 .^ range(-2, 2; length = n)
    f = FFTLog(BesselJKernel(0), r)

    a = @. exp(-(r / 1.0)^2)
    A_single = forward(f, a)

    A_batch = hcat(a, 2 .* a, 0.5 .* a)
    out = forward(f, A_batch)
    ws = FFTLogWorkspace(f, A_batch)
    out_ws = forward(f, A_batch; workspace = ws)
    @test size(out) == (n, 3)
    @test out_ws ≈ out
    @test isapprox(out[:, 1], A_single; rtol = 1e-12)
    @test isapprox(out[:, 2], 2 .* A_single; rtol = 1e-12)
    @test isapprox(out[:, 3], 0.5 .* A_single; rtol = 1e-12)
    @test inverse(f, out_ws, ws) ≈ A_batch rtol=1e-7
    @test_throws DimensionMismatch forward(f, A_batch[:, 1:2], ws)
end
