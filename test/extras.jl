using WarpedRegressors: BoxCox

@testset "extras" begin
    rng = MersenneTwister(123456)

    b = BoxCox(0.1)
    binv = inv(b)

    x = randn(rng)
    y = b(x)

    @test binv(b(x)) ≈ x
    @test b(binv(x)) ≈ x

    @test log(abs(first(Zygote.gradient(b, x)))) ≈ Bijectors.logabsdetjac(b, x)
end
