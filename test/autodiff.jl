using Test
using Distributions
using StableRNGs
using Zygote
using CategoricalArrays
using IntervalSets
using TypedTables

@testset "Autodiff parameter learning" begin
    rng = StableRNG(42)
    leaf = Leaf(Normal(0.0, 1.0), 1)
    spn = SumProductNetwork(leaf, Dict{Int64,Any}(), SPN.ScopeMap((:x,), (Float64,)))

    X = reshape(rand(rng, Normal(2.0, 1.5), 250), :, 1)
    θ0, pm = initial_params(spn)

    ll0 = meanlogpdf(spn, X, θ0, pm)
    θ, pm, _ = fit_params(spn, X; θ0 = θ0, pm = pm, maxiters = 60, lr = 0.05, verbose = false)
    ll1 = meanlogpdf(spn, X, θ, pm)
    @test ll1 > ll0

    spn_fit = with_params(spn, θ, pm)
    @test isfinite(logpdf(spn_fit, [0.0]))
end

@testset "Autodiff handles large Gamma parameters" begin
    leaf = Leaf(Gamma(1056.83242818947, 0.011791060899680282), 1)
    spn = SumProductNetwork(leaf, Dict{Int64,Any}(), SPN.ScopeMap((:x,), (Float64,)))
    X = reshape([14.20797120083023, 12.866329782612004], :, 1)

    θ0, pm = initial_params(spn)
    @test all(isfinite, θ0)
    @test isfinite(meanlogpdf(spn, X, θ0, pm))
    @test isfinite(logpdf(with_params(spn, θ0, pm), [14.20797120083023]))
end

@testset "Autodiff ignores impossible mixture branches" begin
    impossible = SumNode()
    add!(impossible, Leaf(Gamma(2.0, 1.0), 1), log(0.5))
    add!(impossible, Leaf(Gamma(3.0, 1.0), 1), log(0.5))

    root = SumNode()
    add!(root, Leaf(Normal(0.0, 1.0), 1), log(0.5))
    add!(root, impossible, log(0.5))

    spn = SumProductNetwork(root, Dict{Int64,Any}(), SPN.ScopeMap((:x,), (Float64,)))
    X = reshape([-1.0], :, 1)
    θ0, pm = initial_params(spn)
    data = encode_data(spn, X)
    loss(θ) = -meanlogpdf(spn, data, θ, pm; encoded = true)
    g = only(Zygote.gradient(loss, θ0))

    @test isfinite(loss(θ0))
    @test all(isfinite, g)
end

@testset "Autodiff trains interval observations" begin
    leaf = Leaf(Normal(0.0, 1.0), 1)
    spn = SumProductNetwork(leaf, Dict{Int64,Any}(), SPN.ScopeMap((:x,), (Float64,)))
    X = reshape(Any[-0.5..0.5, 1.0..2.0, missing], :, 1)
    θ0, pm = initial_params(spn)
    loss(θ) = -meanlogpdf(spn, X, θ, pm)
    g = only(Zygote.gradient(loss, θ0))

    @test isfinite(loss(θ0))
    @test all(isfinite, g)
    θ, pm, history = fit_params(spn, X; θ0 = θ0, pm = pm, maxiters = 2, verbose = false)
    @test all(isfinite, θ)
    @test all(isfinite, history)
end

@testset "Autodiff trains finite discrete intervals" begin
    leaf = Leaf(Poisson(2.0), 1)
    spn = SumProductNetwork(leaf, Dict{Int64,Any}(), SPN.ScopeMap((:x,), (Int,)))
    X = reshape(Any[1..3, 0..1, missing], :, 1)
    θ0, pm = initial_params(spn)
    loss(θ) = -meanlogpdf(spn, X, θ, pm)
    g = only(Zygote.gradient(loss, θ0))

    @test isfinite(loss(θ0))
    @test all(isfinite, g)
end

@testset "Autodiff trains categorical set observations" begin
    D0 = Table(y = categorical(["A", "B", "C", "A", "B", "A"]))
    spn = learnSPN(D0, 0.2)
    D = Table(y = Any["A", ["A", "B"], missing, "C"])
    θ0, pm = initial_params(spn)
    loss(θ) = -meanlogpdf(spn, D, θ, pm)
    g = only(Zygote.gradient(loss, θ0))

    @test isfinite(loss(θ0))
    @test all(isfinite, g)
end

@testset "Autodiff validates training data" begin
    normal = SumProductNetwork(Leaf(Normal(), 1), Dict{Int64,Any}(), SPN.ScopeMap((:x,), (Float64,)))
    θn, pmn = initial_params(normal)
    @test_throws ArgumentError meanlogpdf(normal, reshape([1.0 2.0], 1, 2), θn, pmn)
    @test_throws ArgumentError meanlogpdf(normal, Table(z = [1.0]), θn, pmn)
    @test_throws ArgumentError meanlogpdf(normal, reshape(Any[[1.0, 2.0]], :, 1), θn, pmn)

    D0 = Table(y = categorical(["A", "B", "A"]))
    cat_spn = learnSPN(D0, 0.2)
    θc, pmc = initial_params(cat_spn)
    @test_throws ArgumentError meanlogpdf(cat_spn, Table(y = ["Z"]), θc, pmc)
    @test_throws ArgumentError meanlogpdf(cat_spn, reshape(Any["A"], :, 1), θc, pmc)

    encoded = encode_data(cat_spn, Table(y = ["A", ["A", "B"]]))
    @test isfinite(meanlogpdf(cat_spn, encoded, θc, pmc; encoded = true))
end
