using Bijectors: Exp
using WarpedRegressors: posterior

@testset "warped_regressor" begin
    @testset "Stheno" begin
        rng = MersenneTwister(123456)

        # Construct a GP.
        N = 11
        s = 0.1
        x = randn(rng, N)
        f = GP(EQ(), GPC())
        fx = f(x, s)

        # Construct a WarpedGP.
        ϕ = Exp()
        ϕf = warp(f, ϕ)
        ϕfx = ϕf(x, s)

        # Ensure that sampling acts heirarchically. Not a phenominal test.
        z = rand(MersenneTwister(1), fx)
        y = ϕ.(z)
        @test y == rand(MersenneTwister(1), ϕfx)

        # Roughly check that the logpdf is correct. This really isn't a great test.
        manual_logpdf = sum(logabsdetjacinv.(Ref(inv(ϕ)), y)) + logpdf(fx, inv(ϕ).(y))
        @test manual_logpdf ≈ logpdf(ϕfx, y)

        # Check that gradients of the lml w.r.t. the parameters can be computed.
        function warped_regressor_logpdf(x, y, s)
            ϕf = warp(GP(EQ(), GPC()), Exp())
            return logpdf(ϕf(x, s), y)
        end
        Zygote.gradient(warped_regressor_logpdf, x, y, s)

        # Check that posterior does the correct thing. This really also isn't a great test.
        manual_posterior = warp(f | (fx ← z), ϕ)
        post = posterior(ϕfx, y)
        @test logpdf(manual_posterior(x), y) == logpdf(post(x), y)
    end
end
