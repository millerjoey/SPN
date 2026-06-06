using Test
using Distributions
using StableRNGs
using Zygote
using CategoricalArrays
using IntervalSets
using TypedTables

function _captured_error(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

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

    θ_batch, _, history_batch = fit_params(
        spn,
        X;
        θ0 = θ0,
        pm = pm,
        maxiters = 60,
        lr = 0.05,
        batch_size = 32,
        rng = StableRNG(7),
        verbose = false,
    )
    @test length(history_batch) == 60
    @test all(isfinite, history_batch)
    @test meanlogpdf(spn, X, θ_batch, pm) > ll0

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

@testset "Autodiff trains with missing observations" begin
    leaf = Leaf(Normal(0.0, 1.0), 1)
    spn = SumProductNetwork(leaf, Dict{Int64,Any}(), SPN.ScopeMap((:x,), (Float64,)))
    X = reshape(Any[missing, 1.0, 2.0, missing], :, 1)
    θ0, pm = initial_params(spn)
    loss(θ) = -meanlogpdf(spn, X, θ, pm)
    g = only(Zygote.gradient(loss, θ0))

    @test isfinite(loss(θ0))
    @test all(isfinite, g)
    θ, pm, history = fit_params(spn, X; θ0 = θ0, pm = pm, maxiters = 2, verbose = false)
    @test all(isfinite, θ)
    @test all(isfinite, history)

    X_missing = reshape(Any[missing, missing], :, 1)
    θ_missing, _, history_missing = fit_params(spn, X_missing; θ0 = θ0, pm = pm, maxiters = 2, verbose = false)
    @test θ_missing == θ0
    @test history_missing == [-0.0, -0.0]
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

@testset "Autodiff uses CDFs for Poisson intervals" begin
    leaf = Leaf(Poisson(2.0), 1)
    spn = SumProductNetwork(leaf, Dict{Int64,Any}(), SPN.ScopeMap((:x,), (Int,)))
    θ0, pm = initial_params(spn)
    dist = Poisson(2.0)

    @test meanlogpdf(spn, reshape(Any[1..3], :, 1), θ0, pm) ≈ log(cdf(dist, 3) - cdf(dist, 0))
    @test meanlogpdf(spn, reshape(Any[-Inf..3], :, 1), θ0, pm) ≈ logcdf(dist, 3)
    @test meanlogpdf(spn, reshape(Any[5..Inf], :, 1), θ0, pm) ≈ logccdf(dist, 4)
    @test meanlogpdf(spn, reshape(Any[0..1_000_000], :, 1), θ0, pm) ≈ 0.0 atol = 1e-12

    loss(θ) = -meanlogpdf(spn, reshape(Any[10..1_000_000], :, 1), θ, pm)
    g = only(Zygote.gradient(loss, θ0))
    @test isfinite(loss(θ0))
    @test all(isfinite, g)
end

@testset "Autodiff trains infinite discrete intervals" begin
    for leaf in (Leaf(Poisson(2.0), 1), Leaf(NegativeBinomial(4.0, 0.35), 1))
        spn = SumProductNetwork(leaf, Dict{Int64,Any}(), SPN.ScopeMap((:x,), (Int,)))
        X = reshape(Any[2..Inf, -Inf..1, 0..Inf, missing], :, 1)
        θ0, pm = initial_params(spn)
        loss(θ) = -meanlogpdf(spn, X, θ, pm)
        g = only(Zygote.gradient(loss, θ0))

        @test isfinite(loss(θ0))
        @test all(isfinite, g)
        θ, pm, history = fit_params(spn, X; θ0 = θ0, pm = pm, maxiters = 2, verbose = false)
        @test all(isfinite, θ)
        @test all(isfinite, history)
    end
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

@testset "Autodiff validates parameter maps" begin
    leaf = Leaf(Normal(), 1)
    spn = SumProductNetwork(leaf, Dict{Int64,Any}(), SPN.ScopeMap((:x,), (Float64,)))
    X = reshape([0.0, 1.0], :, 1)
    θ, pm = initial_params(spn)

    short_θ = θ[1:1]
    @test_throws ArgumentError meanlogpdf(spn, X, short_θ, pm)

    missing_ranges = ParamMap(Dict{UInt128,UnitRange{Int}}(), copy(pm.leaf_kind))
    @test_throws ArgumentError meanlogpdf(spn, X, θ, missing_ranges)

    bad_ranges = copy(pm.ranges)
    bad_ranges[leaf.id] = 1:1
    @test_throws ArgumentError meanlogpdf(spn, X, θ, ParamMap(bad_ranges, copy(pm.leaf_kind)))

    bad_leaf_kind = copy(pm.leaf_kind)
    bad_leaf_kind[leaf.id] = :gamma
    @test_throws ArgumentError meanlogpdf(spn, X, θ, ParamMap(copy(pm.ranges), bad_leaf_kind))
end

@testset "Autodiff reports nonfinite training states" begin
    leaf = Leaf(Normal(), 1)
    spn = SumProductNetwork(leaf, Dict{Int64,Any}(), SPN.ScopeMap((:x,), (Float64,)))
    X = reshape([2.0], :, 1)
    θ0, pm = initial_params(spn)

    bad_θ = copy(θ0)
    bad_θ[1] = NaN
    err = _captured_error() do
        fit_params(spn, X; θ0 = bad_θ, pm = pm, maxiters = 1, verbose = false)
    end
    @test err isa NonFiniteTrainingError
    @test err.stage === :initial_parameters
    @test err.index == 1
    @test err.node_kind === :normal
    @test err.range == pm.ranges[leaf.id]

    err = _captured_error() do
        SPN._assert_gradient([0.0, NaN], pm, 7)
    end
    @test err isa NonFiniteTrainingError
    @test err.stage === :gradient
    @test err.iter == 7
    @test err.index == 2

    gamma_spn = SumProductNetwork(Leaf(Gamma(0.5, 1.0), 1), Dict{Int64,Any}(), SPN.ScopeMap((:x,), (Float64,)))
    err = _captured_error() do
        fit_params(gamma_spn, reshape([0.0], :, 1); maxiters = 1, verbose = false)
    end
    @test err isa NonFiniteTrainingError
    @test err.stage === :loss
    @test err.iter == 1
    @test err.index === nothing

    err = _captured_error() do
        fit_params(spn, X; θ0 = θ0, pm = pm, maxiters = 1, lr = Inf, verbose = false)
    end
    @test err isa NonFiniteTrainingError
    @test err.stage === :updated_parameters
    @test err.iter == 1
    @test err.index !== nothing
end
