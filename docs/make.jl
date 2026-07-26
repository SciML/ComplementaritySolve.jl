using ComplementaritySolve
using Documenter

makedocs(;
    modules = [ComplementaritySolve],
    checkdocs = :public,
    sitename = "ComplementaritySolve.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://docs.sciml.ai/ComplementaritySolve/stable/",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Developer API" => "developer_api.md",
    ],
)

deploydocs(;
    repo = "github.com/SciML/ComplementaritySolve.jl",
    devbranch = "main",
)
