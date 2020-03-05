using Bijectors, Optim, Plots, Random, StatsFuns, Stheno, WarpedRegressors

using WarpedRegressors: InvBoxCox, affine, posterior

# Specify a Warped-GP.
f = GP(kernel(EQ(); l=0.5), GPC())
σ² = 0.01
ϕ = affine(0.0, 1.0) ∘ InvBoxCox(0.5)
ϕ = affine(0.0, 1.0)
ϕ = Bijectors.Shift(1.0)
ϕ = Bijectors.Scale(2.0)
g = warp(f, ϕ)

# Generate some training data.
rng = MersenneTwister(123456)
Ntr = 200
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
    λ = log1pexp(θ[3]) + 1e-6,
)

# Infer σ², l, and s.
function nlml(θ::AbstractVector{<:Real})
    θ = unpack(θ)
    f = GP(kernel(EQ(); l=θ.l), GPC())
    # ϕ = affine(θ.λ_affine...)
    ϕ = Bijectors.Scale(θ.λ)
    g = warp(f, ϕ)
    return -logpdf(g(xtr, θ.σ²), ytr)
end
θ0 = randn(3)
opts = Optim.Options(iterations = 1_000, show_trace=false, extended_trace=false);
results = Optim.optimize(nlml, θ0, NelderMead(), opts)
θ = unpack(results.minimizer)


# Construct posterior Warped GP
f_learned = GP(kernel(EQ(); l=θ.l, s=θ.s), GPC())
g_learned = warp(f_learned, ϕ)
g_posterior = posterior(g_learned(xtr, θ.σ²), ytr)

# Visualise the posterior.
let
    xpr = range(-10.0, 10.0; length=400)
    Ypr = hcat([rand(rng, g_posterior(xpr, θ.σ²)) for _ in 1:500]...)
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
