# Structure Learning Misc. Tests
using Random, Combinatorics, CategoricalArrays
using TypedTables


@testset "Clustering" begin
    Random.seed!(1)
    D = Table(x1=randn(10), x2=randn(10), x3=categorical(rand(["a", "b"], 10)), x4=categorical(rand(["x", "y"], 10)))
    @test cluster(D, 3)==(Bool[1, 0, 0, 0, 0, 0, 1, 1, 1, 1], Bool[0, 1, 1, 1, 1, 1, 0, 0, 0, 0])
end

@testset "Factoring" begin
    Random.seed!(1)
    Vs = [randn(10), randn(10), categorical(rand(["a", "b"], 10)), categorical(rand(["x", "y"], 10))]
    answers = [0.5754008763676006,0.6698153575994181,1.0,0.2864220227778595,0.5224312849615658,0.5981614526835279]
    for (ans,comb) in zip(answers,combinations(1:length(Vs), 2))
        @test indeptest(Vs[comb[1]], Vs[comb[2]]) ≈ ans
    end
    n = 110
    X = [ifelse.(rand(n) .< 0.05, missing, randn(n)*2 .+ 3) ifelse.(rand(n) .< 0.05, missing, rand(Gamma(2,3), n)) ifelse.(rand(n) .< 0.08, missing, rand(Beta(1, 0.1), n))]
    X = [X X[:, 1]+X[:, 2]]
    D = Table((x₁ = X[:, 1], x₂ = X[:, 2], x₃ = X[:, 3], x₄ = X[:, 4]))
    x5 = categorical([ismissing(el) ? missing : (el > 2 ? "A" : (el < 1  ? "B" : "C")) for el in D.x₁])
    x6 = categorical([ismissing(el) ? missing : (el > 10 ? "D" : "E") for el in D.x₄])
    D = Table(D; x₅=x5)
    D = Table(D; x₆=x6)
    H = test_similarity(D, ntest_samps = 110)
    H .= ifelse.(H .< 0.05, 1, 0)
    @test sum(H)==20
    @test factor(D[1:20]) == [[1,5],[2,3,4,6]]
end

@testset "Distribution Fitting" begin
    Random.seed!(2)
    x_cat = categorical(rand(["a", "b"], 10))
    @test fit_dist(x_cat)==Categorical([0.5, 0.5])
    x_pois = [1,2,0,0,1,1,0]
    @test fit_dist(x_pois)==Poisson(mean(x_pois))
    x_norm = rand(Normal(), 30)
    @test fit_dist(x_norm)==fit(Normal, x_norm)
    x_negbin = [1,2,0,0,1,1,8]
    @test fit_dist(x_negbin)==NegativeBinomial(0.5794285714285714, 0.2378048780487805)
    x_gamma = rand(Gamma(), 30)
    @test fit_dist(x_gamma)==Gamma(0.8795818306790851, 1.041271421986373)
end

@testset "Structure Learning" begin
    Random.seed!(1)
    D = Table(x = randn(50), y = categorical(rand(["a", "b", "c"], 50)), z = rand(50))
    spn = learnSPN(D, 0.3)
    @test keys(spn.ScM)==columnnames(D)
    @test spn.ScM.Types==(x=Float64,y=UInt32,z=Float64)
    @test levels(spn.categorical_pool[2])==["a","b","c"]
    @test length(spn.categorical_pool)==1
    @test spn.root isa SPN.ProductNode
    @test length(children(spn.root))==3
end
