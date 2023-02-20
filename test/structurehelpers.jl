# Structure Learning Misc. Tests
using Random, Combinatorics, CategoricalArrays, StableRNGs
using TypedTables


@testset "Clustering" begin
    rng = StableRNG(123)
    D = Table(x1=randn(rng,10), x2=randn(rng,10), x3=categorical(rand(rng,["a", "b"], 10)), x4=categorical(rand(rng,["x", "y"], 10)))
    @test cluster(D, 3)==(Bool[1, 1, 0, 1, 0, 0, 0, 0, 0, 0], Bool[0, 0, 1, 0, 1, 1, 1, 1, 1, 1])
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
    @test sum(H)==20
    @test factor(D, 0.0001) == [[1,5],[2,4,6],[3]]
end

@testset "Distribution Fitting" begin
    rng = StableRNG(1)
    x_cat = categorical(rand(rng, ["a", "b"], 10))
    @test fit_dist(x_cat)==Categorical([0.6000000000000001, 0.4])
    x_pois = [1,2,0,0,1,1,0]
    @test fit_dist(x_pois)==Poisson(mean(x_pois))
    x_norm = rand(rng, Normal(), 30)
    @test fit_dist(x_norm)==fit(Normal, x_norm)
    x_negbin = [1,2,0,0,1,1,8]
    @test fit_dist(x_negbin)==NegativeBinomial(0.5794285714285714, 0.2378048780487805)
    x_gamma = rand(rng, Gamma(), 30)
    @test fit_dist(x_gamma)==Gamma(0.7468273202991, 1.3230179871335725)
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
