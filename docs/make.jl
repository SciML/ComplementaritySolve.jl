using ComplementaritySolve
using Documenter

makedocs(;
    modules = [ComplementaritySolve],
    checkdocs = :exports,
    sitename = "ComplementaritySolve.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://docs.sciml.ai/ComplementaritySolve/stable/",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/SciML/ComplementaritySolve.jl",
    devbranch = "main",
)
