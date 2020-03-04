module WarpedRegressors

using BayesianLinearRegressors, Bijectors, Distributions, Random, Stheno

using Bijectors: AbstractBijector
using Stheno: AbstractGP

export warp, posterior

include("warped_regressor.jl")

end # module
