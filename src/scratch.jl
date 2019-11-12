# TODO: clean up
# Put on Github
# Allow to fit kernel densities as leaves. Can use Distributions package.
# Add way of printing SPN as a mixture model for fun.

using Revise
# Need to activate SPN
using SPN
using IntervalSets, NamedArrays,CategoricalArrays,Distributions
using BenchmarkTools

n = 1000
X = [ifelse.(rand(n) .< 0.05, missing, randn(n)*2 .+ 3) ifelse.(rand(n) .< 0.05, missing, rand(Gamma(2,3), n)) ifelse.(rand(n) .< 0.08, missing, rand(Beta(1, 0.1), n))]

X = [X X[:, 1]+X[:, 2]]
using TypedTables
D = Table((x₁ = X[:, 1], x₂ = X[:, 2], x₃ = X[:, 3], x₄ = X[:, 4]))
x5 = categorical([ismissing(el) ? missing : (el > 2 ? "A" : (el < 1  ? "B" : "C")) for el in D.x₁])
x6 = categorical([ismissing(el) ? missing : (el > 10 ? "D" : "E") for el in D.x₄])
D = Table(D; x₅=x5)
D = Table(D; x₆=x6)

import Juno: @enter,@run,@profiler
# still scopeunion problems
using ProfileView
@time spn = learnSPN(D, 0.3)

@time S = rand(spn, 10000)

cov(S[:, 1:4]), cor(S[:, 1:4])




all_nonmiss = mapreduce(x->.!ismissing.(x), (x,y)->x .& y, columns(D))
D_nonmiss = mapreduce(col->col, hcat, columns(D[all_nonmiss]))

cov(D_nonmiss[:, 1:4]),
cor(D_nonmiss[:, 1:4])




logpdf(spn, [-1, missing, missing, 2, missing, missing]) - logpdf(spn, [-1, missing, missing, missing, missing, missing]) |> exp

@time logpdf(spn, [missing, missing, missing, missing, missing,missing]) |> exp






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
