using FFTLoggin
using Test

@testset "domain checking" begin
    k = BesselJKernel(0.5)
    @test FFTLoggin.isindomain(k, 1.0 + 0im)
    @test !FFTLoggin.isindomain(k, -1.0 + 0im)
    @test !FFTLoggin.isindomain(k, 2.0 + 0im)

    # Shifted
    sk = shift(k, 0.25)
    @test FFTLoggin.isindomain(sk, 0.5 + 0im) ==
          FFTLoggin.isindomain(k, 0.75 + 0im)

    # Derivative shifts strip up
    d = derive(k, 1)
    @test FFTLoggin.isindomain(d, 1.5 + 0im) ==
          FFTLoggin.isindomain(k, 0.5 + 0im)

    # FFTLog with bad bias warns rather than throws
    @test_logs (:warn, r".*outside.*") FFTLog(
        BesselJKernel(0);
        n = 16,
        dlog = 0.1,
        bias = 5.0,
        lowring = false
    )
end
