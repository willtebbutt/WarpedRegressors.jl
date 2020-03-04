using BayesianLinearRegressors, Bijectors, LinearAlgebra, Random, Stheno, Test,
    WarpedRegressors, Zygote

@testset "WarpedRegressors.jl" begin
    include("warped_regressor.jl")
end
