# FFTLoggin.jl

```@raw html
<p align="center">
  <img src="assets/logo-banner.svg" alt="FFTLoggin.jl Logo" width="600"/>
</p>
```

`FFTLoggin.jl` provides FFTLog transforms and kernel abstractions for logarithmically
sampled functions.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/binado/FFTLoggin.jl.git")
```

## Quickstart

```julia
using FFTLoggin

r = 10 .^ range(-2, 2; length = 128)
fftlog = FFTLog(BesselJKernel(0), r; kr = 1.0)

signal = @. exp(-r^2)
transformed = forward(fftlog, signal)
recovered = inverse(fftlog, transformed)
```

## Package Contents

```@contents
Pages = ["index.md", "api.md"]
Depth = 2
```
