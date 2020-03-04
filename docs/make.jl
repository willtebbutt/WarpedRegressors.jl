using Documenter, WarpedRegressors

makedocs(;
    modules=[WarpedRegressors],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/willtebbutt/WarpedRegressors.jl/blob/{commit}{path}#L{line}",
    sitename="WarpedRegressors.jl",
    authors="willtebbutt <wt0881@my.bristol.ac.uk>",
    assets=String[],
)

deploydocs(;
    repo="github.com/willtebbutt/WarpedRegressors.jl",
)
