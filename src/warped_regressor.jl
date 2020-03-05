"""
    WarpedRegressor

Hierarchical model induced by compositing `ϕ` with `f`. Thus for a particular sampled
function `f`, `y = ϕ(f(x))`.
"""
struct WarpedRegressor{Tf, Tϕ<:AbstractBijector}
    f::Tf
    ϕ::Tϕ
end

warp(f, ϕ) = WarpedRegressor(f, ϕ)

# For internal use only. Similar to `FiniteGP`.
struct FiniteWarpedRegressor{Tfx, Tϕ<:AbstractBijector}
    fx::Tfx
    ϕ::Tϕ
end

# Just passes all inputs to the regressor contained in `f`.
(f::WarpedRegressor)(x...) = FiniteWarpedRegressor(f.f(x...), f.ϕ)

# Sample from the warped regressor by warping samples from the regressor.
Random.rand(rng::AbstractRNG, fx::FiniteWarpedRegressor) = fx.ϕ.(rand(rng, fx.fx))

# Compute the logpdf of the warped regressor using the change-of-variables formula.
function Distributions.logpdf(fx::FiniteWarpedRegressor, y::AbstractVector)
    ϕinv = inv(fx.ϕ)
    ladj = sum(logabsdetjacinv.(Ref(ϕinv), y))
    logpdf_latent = logpdf(fx.fx, ϕinv.(y))
    return ladj + logpdf_latent 
end

"""
    posterior(fx::FiniteWarpedRegressor, y::AbstractVector)

Compute the posterior distribution over `f` given observations `y`.
"""
function posterior(fx::FiniteWarpedRegressor{<:Stheno.FiniteGP}, y::AbstractVector)
    return WarpedRegressor((fx.fx.f | (fx.fx ← inv(fx.ϕ).(y))), fx.ϕ)
end

function posterior(
    fx::FiniteWarpedRegressor{<:BayesianLinearRegressors.IndexedBLR},
    y::AbstractVector,
)
    return WarpedRegressor(BayesianLinearRegressors.posterior(fx.fx, inv(fx.ϕ).(y)), fx.ϕ)
end
