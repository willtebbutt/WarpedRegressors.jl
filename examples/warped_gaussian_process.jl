using Bijectors, LinearAlgebra, Optim, Plots, Random, StatsFuns, Stheno, WarpedRegressors

# Specify a Warped-GP.
f = GP(kernel(EQ(); l=2.0, s=0.5), GPC())
σ² = 0.0001
ϕ = Bijectors.Exp()
g = warp(f, ϕ)

# Generate some training data.
rng = MersenneTwister(123456)
Ntr = 35
xtr = 3 * randn(rng, Ntr)
g_xtr = g(xtr, σ²)
ytr = rand(rng, g_xtr)

# Specify parameter handler.
unpack(θ) = (
    σ² = log1pexp(θ[1]) + 1e-6,
    l  = log1pexp(θ[2]) + 1e-6,
    s  = log1pexp(θ[3]) + 1e-6,
)

# Infer σ², l, and s.
function nlml(θ::AbstractVector{<:Real})
    θ = unpack(θ)
    f = GP(kernel(EQ(); l=θ.l, s=θ.s), GPC())
    g = warp(f, ϕ)
    return -logpdf(g(xtr, θ.σ²), ytr)
end
θ0 = randn(3)
results = Optim.optimize(nlml, θ0, NelderMead())
θ = unpack(results.minimizer)

# Construct posterior Warped GP
f_learned = GP(kernel(EQ(); l=θ.l, s=θ.s), GPC())
g_learned = warp(f_learned, ϕ)
g_posterior = posterior(g_learned(xtr, θ.σ²), ytr)

# Visualise the posterior.
let
    xpr = range(-10.0, 10.0; length=250)
    Ypr = hcat([rand(rng, g_posterior(xpr, θ.σ²)) for _ in 1:1000]...)
    median_pr = map(ys->quantile(ys, 0.5), eachrow(Ypr));
    q95_pr = map(ys->quantile(ys, 0.95), eachrow(Ypr));
    q05_pr = map(ys->quantile(ys, 0.05), eachrow(Ypr));

    plotly()
    plt = plot()
    plot!(plt, xpr, Ypr[:, 1:10]; linecolor=:blue, linewidth=0.5, linealpha=0.2, label="")
    plot!(plt, xpr, median_pr; linecolor=:blue, linewidth=1.0, linealpha=1.0, label="")
    plot!(plt, xpr, [median_pr, median_pr];
        linewidth=0.0,
        fillrange=[q05_pr, q95_pr],
        fillalpha=0.3,
        fillcolor=:blue,
        label="",
    )
    scatter!(plt, xtr, ytr; label="", markersize=2)
    display(plt)
end
