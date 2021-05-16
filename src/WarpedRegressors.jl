module WarpedRegressors

using AbstractGPs
using Bijectors
using Random

using AbstractGPs: AbstractGP

include("warped_regressor.jl")
include("extras.jl")

export warp

end # module
