# TODO: clean up
# Put on Github
# Allow to fit kernel densities as leaves. Can use Distributions package.
# Add way of printing SPN as a mixture model for fun.
# look into https://arxiv.org/pdf/1908.03250.pdf
# Need to activate SPN
Pkg.activate(".")
using Revise
using SPN
using IntervalSets, NamedArrays, CategoricalArrays, Distributions
using BenchmarkTools

n = 110
X = [ifelse.(rand(n) .< 0.05, missing, randn(n)*2 .+ 3) ifelse.(rand(n) .< 0.05, missing, rand(Gamma(2,3), n)) ifelse.(rand(n) .< 0.08, missing, rand(Beta(1, 0.1), n))]

X = [X X[:, 1]+X[:, 2]]
using TypedTables
D = Table((x₁ = X[:, 1], x₂ = X[:, 2], x₃ = X[:, 3], x₄ = X[:, 4]))
x5 = categorical([ismissing(el) ? missing : (el > 2 ? "A" : (el < 1  ? "B" : "C")) for el in D.x₁])
x6 = categorical([ismissing(el) ? missing : (el > 10 ? "D" : "E") for el in D.x₄])
D = Table(D; x₅=x5)
D = Table(D; x₆=x6)

import Juno: @enter,@run,@profiler

# Still some issues. Not sure where.
using Random

@time spn = learnSPN(D, 0.3)

@btime rand(spn, 100)
# 705.9 microseconds

@btime rand(spn, 100, [1, missing, missing, missing, "C", missing])
# 945 microseconds

@btime rand(spn, 100, Dict(1=>1, 5=>"C"))
# 879 microseconds


@time rand(spn, 1000)
# 1.047 ms (16642 allocations: 767.83 KiB)


@btime rand(spn, 1000, SPN.AllMissing())
@btime S = rand(spn, 10000)
@time S1 = rand(spn, 10000,  [missing, 2..4, missing, missing, missing, missing])

@time exp.(logpdf(spn, S1))

@time r = rand(spn, 100)
@enter logpdf(spn, r)


cov(S[:, 1:4]), cor(S[:, 1:4])




all_nonmiss = mapreduce(x->.!ismissing.(x), (x,y)->x .& y, columns(D))
D_nonmiss = mapreduce(col->col, hcat, columns(D[all_nonmiss]))

cov(D_nonmiss[:, 1:4]),
cor(D_nonmiss[:, 1:4])


logpdf(spn, [missing, missing, missing, -Inf..Inf, missing, ["E", "D"]]) |> exp
@time logpdf(spn, Dict(:x₁ => -1, :x₅ => ["A", "B", "C"]))
@time logpdf(spn, [-1, missing, missing, missing, missing, missing])

@time logpdf(spn, [-1, missing, 1..4, 2, missing, "E"]) - logpdf(spn, [-1, missing, missing, missing, missing, missing]) |> exp

@btime logpdf(spn, [1..20, missing, missing, missing, missing, missing]) |> exp






using RDatasets
satact = dataset("psych", "sat.act")

filter(r->(r[4]>600) & (10>r[5]>4), RDatasets.datasets())









satact = FlexTable(satact)
satact.Education = categorical(satact.Education)
satact.Gender = categorical(satact.Gender)
satact = Table(satact)
@time spn = learnSPN(satact, 0.3)



# TODO
Make package implementing MFA, use instead of calling R.
Test so it achieves the same results.




###
center_scale(x) = (x .- mean(x))/sqrt(var(x, corrected = false))
x = Table(x1 = randn(100), x2 = randn(100), x3 = randn(100))
x = Table(map(center_scale, columns(x)))
using MixedPCA
U, Δ, Vt = pcamix(x)
