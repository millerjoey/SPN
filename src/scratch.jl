# TODO: clean up
# Put on Github
# Allow to fit kernel densities as leaves. Can use Distributions package.
    # Or maybe chow-liu trees?
    # Or make it build a selective SPN. Fit truncated exponential distrs as leaves.
    #   Then can do Bayesian updating?
# Add way of printing SPN as a mixture model for fun.
# look into https://arxiv.org/pdf/1908.03250.pdf
# Need to activate SPN
Pkg.activate(".")
using Revise
using SPN
using IntervalSets, NamedArrays, CategoricalArrays, Distributions
using BenchmarkTools

n = 110
X = [ifelse.(rand(n) .< 0.05, missing, randn(n)*2 .+ 1) ifelse.(rand(n) .< 0.05, missing, rand(Gamma(2,3), n)) ifelse.(rand(n) .< 0.08, missing, rand(Geometric(0.1), n))]

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

@time spn = learnSPN(D, 0.1)


@time S = rand(spn, 10000, Dict(1=>missing, 2=>1.))
@time S = rand(spn, 10000)
