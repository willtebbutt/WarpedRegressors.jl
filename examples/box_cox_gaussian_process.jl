using Bijectors, Optim, Plots, Random, StatsFuns, Stheno, WarpedRegressors

using WarpedRegressors: InvBoxCox, affine

# Specify a Warped-GP.
f = GP(kernel(EQ(); l=0.5, s=3.0), GPC())
σ² = 0.01
ϕ = InvBoxCox(0.5)
g = warp(f, ϕ)

# Generate some training data.
rng = MersenneTwister(123456)
Ntr = 25
Nte = 200
x = 3 * randn(rng, Ntr + Nte)
xtr = x[1:Ntr]
xte = x[Ntr+1:end]
g_x = g(x, σ²)
y = rand(rng, g_x)
ytr = y[1:Ntr]
yte = y[Ntr+1:end]

# Specify parameter handler.
unpack(θ) = (
    σ² = log1pexp(θ[1]) + 1e-6,
    l  = log1pexp(θ[2]) + 1e-6,
    s  = log1pexp(θ[3]) + 1e-6,
    λ  = log1pexp(θ[4]) + 1e-6,
)

# Infer σ², l, and s.
function nlml(θ::AbstractVector{<:Real})
    θ = unpack(θ)
    f = GP(kernel(EQ(); l=θ.l, s=θ.s), GPC())
    ϕ = InvBoxCox(θ.λ)
    g = warp(f, ϕ)
    return -logpdf(g(xtr, θ.σ²), ytr)
end
θ0 = randn(4)
results = Optim.optimize(nlml, θ0, NelderMead())
θ = unpack(results.minimizer)


# Construct posterior Warped GP
f_learned = GP(kernel(EQ(); l=θ.l, s=θ.s), GPC())
g_learned = warp(f_learned, ϕ)
g_posterior = posterior(g_learned(xtr, θ.σ²), ytr)

# Visualise the posterior.
let
    xpr = range(-10.0, 10.0; length=400)
    Ypr = hcat([rand(rng, g_posterior(xpr, θ.σ²)) for _ in 1:5000]...)
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
    scatter!(plt, xtr, ytr; label="", markersize=2, markercolor=:blue)
    scatter!(plt, xte, yte; label="", markersize=2, markercolor=:black)
    display(plt)
end
