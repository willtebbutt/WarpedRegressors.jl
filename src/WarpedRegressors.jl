module WarpedRegressors

using BayesianLinearRegressors, Bijectors, Distributions, Random, Stheno

using Bijectors: AbstractBijector
using Stheno: AbstractGP

using BayesianLinearRegressors: IndexedBLR
import BayesianLinearRegressors: posterior

export warp

include("warped_regressor.jl")
include("extras.jl")

end # module
