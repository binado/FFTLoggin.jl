using FFTLoggin
using Test

@testset "Batched signals (Matrix columns)" begin
    n = 128
    r = 10 .^ range(-2, 2; length = n)
    f = FFTLog(BesselJKernel(0), r)

    a = @. exp(-(r / 1.0)^2)
    A_single = forward(a, f)

    A_batch = hcat(a, 2 .* a, 0.5 .* a)
    out = forward(A_batch, f)
    @test size(out) == (n, 3)
    @test isapprox(out[:, 1], A_single; rtol = 1e-12)
    @test isapprox(out[:, 2], 2 .* A_single; rtol = 1e-12)
    @test isapprox(out[:, 3], 0.5 .* A_single; rtol = 1e-12)
end
