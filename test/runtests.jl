using AbstractGPs
using BayesianLinearRegressors
using Bijectors
using LinearAlgebra
using Random
using Test
using WarpedRegressors
using Zygote

@testset "WarpedRegressors.jl" begin
    include("warped_regressor.jl")
    include("extras.jl")
end
