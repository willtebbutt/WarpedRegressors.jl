# This collection of functionalities should probably be moved to a better home.

"""
    BoxCox

See "Learning non-Gaussian Time Series using the Box-Cox Gaussian Process".
"""
struct BoxCox{Tλ<:Real} <: Bijector{0}
    λ::Tλ
end

(b::BoxCox)(x) = (sign(x) * abs(x)^b.λ - 1) / b.λ
Bijectors.inv(b::BoxCox) = InvBoxCox(b.λ)
Bijectors.logabsdetjac(b::BoxCox, x) = (b.λ - 1) *  log(abs(x))

"""
    InvBoxCox

See "Learning non-Gaussian Time Series using the Box-Cox Gaussian Process".
"""
struct InvBoxCox{Tλ<:Real} <: Bijector{0}
    λ::Tλ
end

(b::InvBoxCox)(y) = sign(b.λ * y + 1) * abs(b.λ * y + 1)^(1 / b.λ)
Bijectors.inv(b::InvBoxCox) = BoxCox(b.λ)
Bijectors.logabsdetjac(b::InvBoxCox, y) = Bijectors.logabsdetjac(inv(b), inv(b)(y))

affine(a, b) = Bijectors.Shift(a) ∘ Bijectors.Scale(b)
