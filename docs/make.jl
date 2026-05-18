using Documenter
using FFTLoggin

makedocs(
    sitename = "FFTLoggin.jl",
    modules = [FFTLoggin],
    checkdocs = :none,
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
    format = Documenter.HTML(
        edit_link = "main",
        repolink = "https://github.com/binado/FFTLoggin.jl",
        inventory_version = "0.1.0-DEV",
    ),
    remotes = nothing,
)

deploydocs(
    repo = "github.com/binado/FFTLoggin.jl.git",
    devbranch = "main",
)
