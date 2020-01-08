export logpdf

# Make logpdf only work on matrices? Will it be faster?
function logpdf(SPN::SumProductNetwork, X::AbstractMatrix)
    lps = Vector{Float64}(undef, size(X, 1))
    for i in 1:size(X, 1)
        lps[i] = logpdf(SPN, convert(Vector{Any}, X[i, :]))
    end
    return(lps)
end

"""
Needs query as dict with columnnames as symbols or
"""
logpdf(SPN::SumProductNetwork, query::AbstractDict) = logpdf(SPN, queryfromdict(SPN, query))

function logpdf(SPN::SumProductNetwork, x::AbstractVector)
    for (scope,pool) in SPN.categorical_pool
        x[scope] isa Missing ? continue : nothing
        if isa(x[scope], AbstractVector)
            x[scope] = [pool.invindex[v] for v in x[scope]]
        else
            x[scope] = pool.invindex[x[scope]]
        end
    end
    logpdf(SPN.root, x)
end

function logpdf(N::SumNode, x::AbstractVector)
    p = 0.
    lps = [logpdf(c, x) for c in children(N)]
    m = lps[argmax(lps)]
    for (lw,lp) in zip(N.logweights,lps)
        p += exp(lw + lp - m)
    end
    lp = log(p) + m
    return isfinite(lp) ? lp : -Inf
end

function logpdf(N::ProductNode, x::AbstractVector)
    lp = 0.
    for c in N.children
        lp += logpdf(c, x)
    end
    return(sum(lp))
end

logpdf(N::Leaf, x::AbstractVector) = _logpdf(N, x[N.scope])

_logpdf(N::Leaf, x::Interval{:closed,:closed}) = log(cdf(N.dist, x.right) - cdf(N.dist, x.left-eps(Float64)))
_logpdf(N::Leaf, x::Interval{:open,:open}) = log(cdf(N.dist, x.right-eps(Float64)) - cdf(N.dist, x.left))
_logpdf(N::Leaf, x::Interval{:open,:closed}) = log(cdf(N.dist, x.right) - cdf(N.dist, x.left))
_logpdf(N::Leaf, x::Interval{:closed,:open}) = log(cdf(N.dist, x.right - eps(Float64)) - cdf(N.dist, x.left - eps(Float64)))
_logpdf(N::Leaf, x::Missing) = 0.
_logpdf(N::Leaf, x::Real) = logpdf(N.dist, x)

_logpdf(N::Leaf{<:DiscreteNonParametric}, x::AbstractVector) = log(sum(pdf.(N.dist, x)))
