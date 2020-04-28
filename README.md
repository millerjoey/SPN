# SPN

Structure learning, sampling, conditional sampling, and conditional log pdfs for sum-product networks I made to help a work project. Compatible with categorical data, missing data.

Shout out to [Martin Trapp](https://github.com/trappmartin) whose SumProductNetwork package inspired me to try to use AD for weight and leaf-distribution parameter learning! I failed, but added some other features...

```julia
using SPN
using IntervalSets, NamedArrays, CategoricalArrays, Distributions
using BenchmarkTools, TypedTables

n = 110
X = [ifelse.(rand(n) .< 0.05, missing, randn(n)*2 .+ 1) ifelse.(rand(n) .< 0.05, missing, rand(Gamma(2,3), n)) ifelse.(rand(n) .< 0.08, missing, rand(Geometric(0.1), n))]

X = [X X[:, 1]+X[:, 2]]
using TypedTables
D = Table((x₁ = X[:, 1], x₂ = X[:, 2], x₃ = X[:, 3], x₄ = X[:, 4]))
x5 = categorical([ismissing(el) ? missing : (el > 2 ? "A" : (el < 1  ? "B" : "C")) for el in D.x₁])
x6 = categorical([ismissing(el) ? missing : (el > 10 ? "D" : "E") for el in D.x₄])
D = Table(D; x₅=x5)
D = Table(D; x₆=x6)

spn = learnSPN(D, 0.1) # Structure learning.
```

## Sampling
Here's the syntax for sampling from the learned distribution:

```julia
one_samp = rand(spn)
many_samps = rand(spn, 10000)

conditional_samps = rand(spn, 10000, Dict(1=>-2., 3=>1.)) # samples from P(X2,X4,X5,X6|X1=-2,X3=1)
rand(spn, 10000, [-2., missing, 1., missing, missing, missing]) # alt syntax
```

You can also sample conditional on categorical values, or multiple:
```julia
rand(spn, 1000, [-2., missing, 1., missing, ["B", "C"], missing]) # samples from P(X2,X4,X6 | X1=-2, X3=1, X5∈{"B","C"})
```

And intervals
```julia
rand(spn, 1000, [-2., 1..2, 1., missing, ["B", "C"], missing]) # samples from P(X4,X6 | X1=-2, X2∈[1,2], X3=1, X5∈{"B","C"})
```

## Evaluating Densities
The same syntax works as with sampling:
```julia
logpdf(spn, many_samps)

logpdf(spn, [-2..Inf, 1..2, missing, missing, ["B", "C"], missing]) # log "density" of P(X4,X6 | X1>-2, X2∈[1,2], X5∈{"B","C"})
```

Beware when trying to do crazy stuff like this:
```julia
logpdf(spn, [-2..Inf, 1, missing, missing, missing, missing]) > logpdf(spn, [-2..Inf, 1..1.01, missing, missing, missing, missing])
```
Right side is a probability, the left is not.

Since it's an SPN, we get any conditional probabilities of course
```julia
exp(logpdf(spn, Dict(1=>2, 6=>"E")) - logpdf(spn, Dict(1=>2))) # P(X6="E" | X1 = 2)
```
