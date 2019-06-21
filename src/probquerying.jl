export logpdf

function logpdf(SPN::SumProductNetwork, X::AbstractMatrix)
    lps = Vector{Float64}(undef, size(X, 1))
    for i in 1:size(X, 1)
        lps[i] = logpdf(SPN, view(X, i, :))
    end
    return(lps)
end

function logpdf(SPN::SumProductNetwork, x::AbstractVector)
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

logpdf(N::Leaf, x::AbstractVector) = logpdf(N, x[N.scope])

logpdf(N::Leaf, x::Interval{:closed,:closed}) = log(cdf(N.dist, x.right) - cdf(N.dist, x.left-eps(Float64)))
logpdf(N::Leaf, x::Interval{:open,:open}) = log(cdf(N.dist, x.right-eps(Float64)) - cdf(N.dist, x.left))
logpdf(N::Leaf, x::Interval{:open,:closed}) = log(cdf(N.dist, x.right) - cdf(N.dist, x.left))
logpdf(N::Leaf, x::Interval{:closed,:open}) = log(cdf(N.dist, x.right - eps(Float64)) - cdf(N.dist, x.left - eps(Float64)))
logpdf(N::Leaf, x::Missing) = 0.
logpdf(N::Leaf, x::Real) = isnan(x) ? 0. : logpdf(N.dist, x)
