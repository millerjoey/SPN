# Structure Learning Misc. Tests
using Random, Combinatorics, CategoricalArrays, StableRNGs, Statistics, Distributions
using TypedTables


@testset "Clustering" begin
    rng = StableRNG(123)
    D = Table(x1=randn(rng,10), x2=randn(rng,10), x3=categorical(rand(rng,["a", "b"], 10)), x4=categorical(rand(rng,["x", "y"], 10)))
    @test length(cluster(D, 3))==2
end

@testset "Missing-value structure preprocessing" begin
    D = Table(
        x = Union{Missing,Float64}[1.0, missing, 3.0, 4.0],
        y = categorical(Union{Missing,String}["a", missing, "b", "a"]),
    )
    ready = SPN.replace_missings(D)

    @test !any(ismissing, ready.x)
    @test !any(ismissing, ready.y)
    @test ready.x[2] == mean(skipmissing(D.x))
    @test !("?" in levels(ready.y))
    @test levels(ready.y) == ["a", "b"]

    sparse = Table(x = Union{Missing,Float64}[1.0, missing], y = Union{Missing,Float64}[missing, 2.0])
    H = test_similarity(sparse)
    @test H[:x, :y] == 1.0

    one_level = categorical(["a", "a", "a"])
    @test indeptest([1.0, 2.0, 3.0], one_level) == 1.0

    all_missing = Table(x = Union{Missing,Float64}[missing, missing])
    @test_throws ArgumentError learnSPN(all_missing, 0.2)
end

@testset "Factoring" begin
    Vs =  [[0.7057204120132424, -0.35308993755882484, 1.4205113346457332, -1.3213660947891328, 0.5808035073416284, -0.11591044168813393, 0.057478142000570435, 0.9055784687894782, -1.1586237091408333, -0.4306733920234955],
    [-2.152516417497302, -0.2050768883565745, 0.46035190687750427, -0.6173087288142342, -0.08238002282037522, -0.8020256421119435, 0.21251592404302283, 2.4315300329382756, -1.6711462073742698, 0.7662299413471073],
    categorical(["a", "a", "b", "a", "b", "a", "b", "a", "a", "a"]),
    categorical(["y", "y", "y", "x", "x", "x", "x", "y", "x", "y"])]
    answers = [0.9386461426590393,0.13847671503211745,0.17452534056858324,0.30505885926167975,0.2505920506856841,0.49015296041582523]
    for (ans,comb) in zip(answers,combinations(1:length(Vs), 2))
        @test indeptest(Vs[comb[1]], Vs[comb[2]]) ≈ ans
    end
    rng = StableRNG(1)
    n = 110
    X = [ifelse.(rand(rng, n) .< 0.05, missing, randn(rng, n)*2 .+ 3) ifelse.(rand(rng, n) .< 0.05, missing, rand(rng, Gamma(2,3), n)) ifelse.(rand(n) .< 0.08, missing, rand(rng, Beta(1, 0.1), n))]
    X = [X X[:, 1]+X[:, 2]]
    D = Table((x₁ = X[:, 1], x₂ = X[:, 2], x₃ = X[:, 3], x₄ = X[:, 4]))
    x5 = categorical([ismissing(el) ? missing : (el > 2 ? "A" : (el < 1  ? "B" : "C")) for el in D.x₁])
    x6 = categorical([ismissing(el) ? missing : (el > 10 ? "D" : "E") for el in D.x₄])
    D = Table(D; x₅=x5)
    D = Table(D; x₆=x6)
    H = test_similarity(D, ntest_samps = 110)
    H .= ifelse.(H .< 0.05, 1, 0)
    @test 10 <= sum(H) <= 40
    @test factor(D, 0.0001) == [[1,5],[2,4,6],[3]]
end

@testset "Distribution Fitting" begin
    rng = StableRNG(1)
    x_cat = categorical(rand(rng, ["a", "b"], 10))
    @test probs(fit_dist(x_cat)) ≈ [0.6, 0.4]
    x_pois = [1,2,0,0,1,1,0]
    @test fit_dist(x_pois)==Poisson(mean(x_pois))
    x_norm = rand(rng, Normal(), 30)
    @test fit_dist(x_norm)==fit(Normal, x_norm)
    x_negbin = [1,2,0,0,1,1,8]
    x̄, σ² = mean(x_negbin), var(x_negbin)
    p = x̄/σ²
    r = x̄^2/(σ²-x̄)
    @test fit_dist(x_negbin)==NegativeBinomial(r, p)
    x_gamma = rand(rng, Gamma(), 30)
    @test fit_dist(x_gamma)==fit(Gamma, x_gamma)

    positive_integer_looking = [1.0, 2.0, 3.0]
    @test fit_dist(positive_integer_looking; numeric_kind = :continuous) == fit(Gamma, positive_integer_looking)
    @test fit_dist(positive_integer_looking; numeric_kind = :continuous, allow_gamma = false) == fit(Normal, positive_integer_looking)

    root = SumNode()
    global_data = Table(x = [0.0, 1.0, 2.5, 3.0])
    subset = Table(x = [1.0, 2.0, 3.0])
    ctx = SPN._learning_context(global_data)
    SPN.add_univariate_leaf!(root, subset, SPN.ScopeMap((:x,), (Float64,)), 1.0; ctx = ctx)
    @test only(children(root)).dist == fit(Normal, subset.x)
end

@testset "Structure Learning" begin
    rng = StableRNG(1)
    D = Table(x = randn(rng,50), y = categorical(rand(rng,["a", "b", "c"], 50)), z = rand(rng,50))
    spn = learnSPN(D, 0.3)
    @test keys(spn.ScM)==columnnames(D)
    @test spn.ScM.Types==(x=Float64,y=UInt32,z=Float64)
    @test levels(spn.categorical_pool[2])==["a","b","c"]
    @test length(spn.categorical_pool)==1
    @test spn.root isa SPN.SumNode
    @test length(children(spn.root))==2
end
