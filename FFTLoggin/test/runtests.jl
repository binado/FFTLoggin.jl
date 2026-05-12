using FFTLoggin
using Test

@testset "FFTLoggin.jl" begin
    include("test_kernels.jl")
    include("test_grid.jl")
    include("test_domain_checking.jl")
    include("test_fftlog.jl")
    include("test_identity.jl")
    include("test_batched.jl")
    include("test_fortran_bench.jl")
end
