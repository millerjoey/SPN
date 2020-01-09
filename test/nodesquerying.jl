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

@testset "Sampling" begin
    Random.seed!(1)
    samps = Dict(1=>Union{Float64, Missing}[missing, missing, missing, missing, missing],
    2=>Union{Float64}[2.5, 2.5, 2.5, 2.5, 2.5])
    SPN.condsamp!(samps, s, 1:5, [missing, 2.5])
    @test sum(samps[1]) == 2.967157950250344
    @test sum(samps[2]) == 2.5*5
end

@testset "Logpdfs" begin
    @test exp(SPN.logpdf(s, SPN.AllMissing())) == 1.
    @test SPN.logpdf(s, [2, 3]) == logpdf(Normal(), 2) + logpdf(Gamma(), 3)
end
