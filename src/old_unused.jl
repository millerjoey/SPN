# Non-mutating (recompile) update to SPN with parameters θ
function replace(SPN::SumProductNetwork, θ)
    parmap = SPN.parmap
    root = replace(SPN.root, θ, SPN.parmap)
    return CompiledSPN(root, parmap)
end

function replace(N::SumNode, θ, parmap)
    children = Tuple(replace(c, θ, parmap) for c in N.children)
    ϕ = θ[parmap[N.id]]
    return SumNode(ϕ/sum(ϕ), children, N.scope, N.id)
end

function replace(N::ProductNode, θ, parmap)
    children = Tuple(replace(c, θ, parmap) for c in N.children)
    return ProductNode(children, N.scope, N.id)
end

# NB
export NB,rand,params,logbinom,logpdf
# Negative Binomial Functions AD-compatible

# Need a new Negative Binomial so I can subtype logpdf so it can be differentiated.
struct NB{T<:Real} <: DiscreteUnivariateDistribution
    r::T
    p::T
end

# Need a pdf of Negative Binomial and to implement logpdf for it.
function logbinomial(n,k)
    s = 0.
    if k ≥ n
        return(0.)
    end
    k = min(k, n-k)
    for i in n:-1:(n-k+1)
        s += log(i)
    end
    for i in 1:k
        s -= log(i)
    end
    s
end

function logpdf(D::NB, k::Integer) # Need to use this to override. Maybe just make new struct to subtype NegativeBinomial so it uses this. Or add method for Duals/trackers whatever.
    r,p = params(D)
    return logbinomial(k + r - 1, k) + r*log(p) + k*log(1-p)
end

function rand(D::NB)
    lambda = rand(Gamma(D.r, (1-D.p)/D.p))
    return rand(Poisson(lambda))
end

params(D::NB) = (D.r, D.p)
