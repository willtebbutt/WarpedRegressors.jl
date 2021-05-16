"""
    WarpedGP

Hierarchical model induced by compositing `ϕ` with `f`. Thus for a particular sampled
function `f`, `y = ϕ(f(x))`.

This subtypes GP, but it's not really a GP, so it shoudn't really do this. It's helpful to
do this because it means that we can utilise FiniteGPs.
"""
struct WarpedGP{Tf<:AbstractGP, Tϕ<:Bijectors.AbstractBijector} <: AbstractGP
    f::Tf
    ϕ::Tϕ
end

warp(f, ϕ) = WarpedGP(f, ϕ)

const FiniteWarpedGP = AbstractGPs.FiniteGP{<:WarpedGP}

build_finite_gp(fx::FiniteWarpedGP) = fx.f.f(fx.x, fx.Σy)

get_ϕ(fx::FiniteWarpedGP) = fx.f.ϕ

# Sample from the warped regressor by warping samples from the regressor.
function AbstractGPs.rand(rng::AbstractRNG, fx::FiniteWarpedGP)
    samples = rand(rng, build_finite_gp(fx))
    return get_ϕ(fx).(samples)
end

# Compute the logpdf of the warped regressor using the change-of-variables formula.
function AbstractGPs.logpdf(fx::FiniteWarpedGP, y::AbstractVector{<:Real})
    ϕinv = inv(get_ϕ(fx))
    ladj = sum(logabsdetjac.(Ref(ϕinv), y))
    logpdf_latent = logpdf(build_finite_gp(fx), ϕinv.(y))
    return ladj + logpdf_latent 
end

# The posterior over a WarpedGP is another WarpedGP.
function AbstractGPs.posterior(fx::FiniteWarpedGP, y::AbstractVector{<:Real})
    ϕ = get_ϕ(fx)
    return WarpedGP(posterior(build_finite_gp(fx), inv(ϕ).(y)), ϕ)
end
