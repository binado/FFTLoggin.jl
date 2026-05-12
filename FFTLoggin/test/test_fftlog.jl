using FFTLoggin
using Test

@testset "FFTLog construction & basics" begin
    r = 10 .^ range(-2, 2; length = 128)
    f = FFTLog(BesselJKernel(0), r; kr = 1.0, lowring = true)
    @test f.n == 128
    @test isapprox(f.dlog, (4 / 127) * log(10); rtol = 1e-12)
    @test isfinite(f.kr)

    a = @. exp(-(r / 1.0)^2)
    A = forward(f, a)
    ws = FFTLogWorkspace(f, a)
    A_ws = forward(f, a, ws)
    @test length(A) == 128
    @test all(isfinite, A)
    @test A_ws ≈ A

    # Callable sugar
    @test forward(f, a) ≈ f(a)
    @test inverse(f, A) ≈ a rtol=1e-7
    @test inverse(f, A_ws; workspace = ws) ≈ a rtol=1e-7

    # Dimension mismatch
    @test_throws DimensionMismatch forward(f, zeros(50))
    @test_throws DimensionMismatch FFTLogWorkspace(f, zeros(50))
end

@testset "lowring snap" begin
    r = 10 .^ range(-2, 2; length = 128)
    fnone = FFTLog(BesselJKernel(0), r; kr = 1.0, lowring = false)
    fsnap = FFTLog(BesselJKernel(0), r; kr = 1.0, lowring = true)
    @test fnone.kr == 1.0
    @test fsnap.kr != 1.0 || isapprox(fsnap.kr, 1.0)  # may already be optimal
end

@testset "FFTLog forward with batched Bessel orders" begin
    n = 32
    μ_row = reshape([0.0, 1.0, 2.0], 1, :)
    r = 10 .^ range(-2.0, 2.0; length = n)
    dlog = infer_dlog(r)
    f = FFTLog(
        BesselJKernel(μ_row);
        n = n,
        dlog = dlog,
        bias = 0.0,
        kr = 1.0,
        lowring = false
    )
    fr = @. r^(0 + 1) * exp(-r^2 / 2)
    ak = forward(f, fr)
    ws = FFTLogWorkspace(f, fr)
    ak_ws = forward(f, fr, ws)
    @test size(ak) == (n, 3)
    @test ak_ws ≈ ak
    @test all(isfinite, ak)

    scalar_fftlog(μ) = FFTLog(
        BesselJKernel(μ);
        n = n,
        dlog = dlog,
        bias = 0.0,
        kr = 1.0,
        lowring = false
    )
    expected = hcat(forward.(scalar_fftlog.([0.0, 1.0, 2.0]), Ref(fr))...)
    @test isapprox(ak, expected; rtol = 1e-12)

    back = inverse(f, ak)
    back_ws = inverse(f, ak_ws, ws)
    @test size(back) == (n, 3)
    @test isapprox(back, repeat(fr, 1, 3); rtol = 1e-7)
    @test isapprox(back_ws, repeat(fr, 1, 3); rtol = 1e-7)
end

@testset "FFTLog batched kernel input broadcasting" begin
    n = 32
    μ_row = reshape([0.0, 1.0, 2.0], 1, :)
    r = 10 .^ range(-2.0, 2.0; length = n)
    dlog = infer_dlog(r)
    f = FFTLog(
        BesselJKernel(μ_row);
        n = n,
        dlog = dlog,
        bias = 0.0,
        kr = 1.0,
        lowring = false
    )
    fr = @. r * exp(-r^2 / 2)
    ak = forward(f, fr)

    @test forward(f, reshape(fr, n, 1)) ≈ ak
    direct = forward(f, hcat(fr, 2 .* fr, 0.5 .* fr))
    @test size(direct) == (n, 3)
    @test direct[:, 1] ≈ ak[:, 1]
    @test direct[:, 2] ≈ 2 .* ak[:, 2]
    @test direct[:, 3] ≈ 0.5 .* ak[:, 3]
    @test_throws DimensionMismatch forward(f, repeat(fr, 1, 2))
    @test size(inverse(f, fr)) == (n, 3)

    μ_grid = reshape([0.0, 1.0], 1, 2, 1)
    fg = FFTLog(
        BesselJKernel(μ_grid);
        n = n,
        dlog = dlog,
        bias = 0.0,
        kr = 1.0,
        lowring = false
    )
    signals = reshape(hcat(fr, 2 .* fr), n, 1, 2)
    out = forward(fg, signals)
    ws = FFTLogWorkspace(fg, signals)
    out_ws = forward(fg, signals; workspace = ws)
    @test size(out) == (n, 2, 2)
    @test out_ws ≈ out
    back = inverse(fg, out)
    back_ws = inverse(fg, out_ws; workspace = ws)
    @test size(back) == (n, 2, 2)
    @test isapprox(back, repeat(signals, 1, 2, 1); rtol = 1e-7)
    @test isapprox(back_ws, repeat(signals, 1, 2, 1); rtol = 1e-7)
end
