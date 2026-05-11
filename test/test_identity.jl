using FFTLoggin
using Test

@testset "Roundtrip identity" begin
    for n in (64, 128, 129)
        r = 10 .^ range(-2, 2; length = n)
        for bias in (0.0, -0.5)
            for kr in (0.5, 1.0)
                f = FFTLog(BesselJKernel(0), r;
                           bias = bias, kr = kr, lowring = true)
                a = @. exp(-(r / 1.0)^2)
                A = forward(f, a)
                a_back = inverse(f, A)
                @test isapprox(a, a_back; atol = 1e-10, rtol = 1e-7)
            end
        end
    end
end
