using Distributions,Random

p = ProductNode()
s = SumNode()
l = Leaf(Normal(), 1)

@testset "Adding, rescoping" begin
    add!(p, l)
    add!(s, p, 0)
    @test scope(p)==scope(s)==[1]
    @test children(s)[1]==p
    add!(p, Leaf(Gamma(), 2))
    @test scope(p)==[1,2]
    rescope!(s)
    @test scope(s)==[1,2]
end

spn = SumProductNetwork(s, Dict{Int64,Any}(), SPN.ScopeMap((:x,:y),(Float64,Float64)))
@testset "Sampling" begin
    Random.seed!(1)
    samps = rand(spn, 10, [missing, 2.5])
    @test sum(samps[:, 1]) == 0.10519496643053128
    @test sum(samps[:, 2]) == 2.5*10
end

@testset "Logpdfs" begin
    @test exp(logpdf(spn, [missing, missing])) == 1.
    @test logpdf(spn, [2, 3]) == logpdf(Normal(), 2) + logpdf(Gamma(), 3)
end
