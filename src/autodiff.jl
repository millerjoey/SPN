export ParamMap, EncodedData
export encode_data, initial_params, meanlogpdf
export fit_params, fit_parameters, with_params

using LogExpFunctions
using Optimisers
using SpecialFunctions
using Zygote
using ChainRulesCore

const _POS_EPS = 1e-6

@inline _softplus(x) = LogExpFunctions.log1pexp(x)
@inline _sigmoid(x) = inv(one(x) + exp(-x))
@inline _safe_prob(p̃) = _POS_EPS + (1 - 2 * _POS_EPS) * _sigmoid(p̃)
@inline _invsoftplus(y) = y > 20 ? y : log(expm1(y))

function _logsumexp(xs)
    m = maximum(xs)
    m == -Inf && return -Inf
    m == Inf && return Inf
    return LogExpFunctions.logsumexp(xs)
end

Zygote.@adjoint function _logsumexp(xs::AbstractVector)
    y = _logsumexp(xs)
    function pullback(ȳ)
        if isfinite(y)
            return (ȳ .* exp.(xs .- y),)
        end
        return (fill(zero(ȳ), length(xs)),)
    end
    return y, pullback
end

struct ParamMap
    ranges::Dict{UInt128,UnitRange{Int}}
    leaf_kind::Dict{UInt128,Symbol}
end

struct EncodedData{C}
    cols::C
    nrows::Int
end

@inline _get(data::EncodedData, row::Int, col::Int) = data.cols[col][row]

function encode_data(spn::SumProductNetwork, X::AbstractMatrix)
    cols = [view(X, :, j) for j in 1:size(X, 2)]
    return EncodedData(cols, size(X, 1))
end

function encode_data(spn::SumProductNetwork, X::Table)
    raw_cols = collect(columns(X))
    cols = AbstractVector[_encode_column(spn, raw_cols[j], j) for j in eachindex(raw_cols)]
    nrows = isempty(cols) ? 0 : length(cols[1])
    return EncodedData(cols, nrows)
end

function _encode_column(spn::SumProductNetwork, col, j::Int)
    if haskey(spn.categorical_pool, j)
        pool = spn.categorical_pool[j]
        if eltype(col) <: UInt32 || eltype(col) <: Union{Missing,UInt32}
            return col
        end
        return Any[_encode_categorical_observation(pool, col[i]) for i in eachindex(col)]
    end
    return col
end

_is_set_observation(x) = x isa AbstractVector || x isa Tuple || x isa AbstractSet

function _encode_categorical_observation(pool, v)
    ismissing(v) && return missing
    if _is_set_observation(v)
        return [_encode_categorical_value(pool, el) for el in v]
    end
    return _encode_categorical_value(pool, v)
end

function _encode_categorical_value(pool, v)
    v isa UInt32 && return v
    v isa CategoricalValue && return pool.invindex[String(v)]
    v isa AbstractString && return pool.invindex[v]
    return pool.invindex[string(v)]
end

function initial_params(spn::SumProductNetwork)
    θ = Float64[]
    ranges = Dict{UInt128,UnitRange{Int}}()
    leaf_kind = Dict{UInt128,Symbol}()
    pos = Ref(1)

    function visit(n::Node)
        if n isa SumNode
            k = length(children(n))
            r = pos[]:(pos[] + k - 1)
            ranges[n.id] = r
            append!(θ, n.logweights)
            pos[] += k
            for c in children(n)
                visit(c)
            end
        elseif n isa ProductNode
            for c in children(n)
                visit(c)
            end
        elseif n isa Leaf
            kind, ϕ0 = _leaf_unconstrained(spn, n)
            leaf_kind[n.id] = kind
            r = pos[]:(pos[] + length(ϕ0) - 1)
            ranges[n.id] = r
            append!(θ, ϕ0)
            pos[] += length(ϕ0)
        else
            error("Unsupported node type: $(typeof(n))")
        end
    end

    visit(spn.root)
    return θ, ParamMap(ranges, leaf_kind)
end

function _leaf_unconstrained(::SumProductNetwork, n::Leaf{<:Normal})
    d = n.dist
    σ = max(d.σ, _POS_EPS)
    return :normal, [d.μ, _invsoftplus(max(σ - _POS_EPS, _POS_EPS))]
end

function _leaf_unconstrained(::SumProductNetwork, n::Leaf{<:Gamma})
    d = n.dist
    α, θ = Distributions.params(d)
    α = max(α, _POS_EPS)
    θ = max(θ, _POS_EPS)
    return :gamma, [_invsoftplus(max(α - _POS_EPS, _POS_EPS)), _invsoftplus(max(θ - _POS_EPS, _POS_EPS))]
end

function _leaf_unconstrained(::SumProductNetwork, n::Leaf{<:Poisson})
    d = n.dist
    λ = max(d.λ, _POS_EPS)
    return :poisson, [_invsoftplus(max(λ - _POS_EPS, _POS_EPS))]
end

function _leaf_unconstrained(::SumProductNetwork, n::Leaf{<:NegativeBinomial})
    d = n.dist
    r, p = d.r, d.p
    r = max(r, _POS_EPS)
    p = clamp(p, _POS_EPS, 1 - _POS_EPS)
    return :negbin, [_invsoftplus(max(r - _POS_EPS, _POS_EPS)), log(p / (1 - p))]
end

function _leaf_unconstrained(spn::SumProductNetwork, n::Leaf{<:Categorical})
    d = n.dist
    # Important: a categorical leaf may have seen only a subset of global levels, yielding probs(d) length 1..K.
    # We expand to the full feature cardinality so scoring/training doesn't BoundsError on unseen categories.
    K_global = haskey(spn.categorical_pool, n.scope) ? length(spn.categorical_pool[n.scope].levels) : length(probs(d))
    return :categorical, _categorical_logits_padded(d, K_global)
end

function _categorical_logits_padded(d::Categorical, K::Int)
    ps = probs(d)
    logits = fill(log(_POS_EPS), K)
    for i in 1:min(length(ps), K)
        logits[i] = log(max(ps[i], _POS_EPS))
    end
    return logits
end

function meanlogpdf(spn::SumProductNetwork, X, θ::AbstractVector, pm::ParamMap; encoded::Bool = false)
    data = encoded ? X : ChainRulesCore.ignore_derivatives() do
        encode_data(spn, X)
    end
    n = data.nrows
    n == 0 && return -Inf
    s = zero(eltype(θ))
    @inbounds for i in 1:n
        s += _logpdf_ad(spn.root, data, i, θ, pm)
    end
    return s / n
end

function _logpdf_ad(n::SumNode, data::EncodedData, row::Int, θ::AbstractVector, pm::ParamMap)
    logits = @view θ[pm.ranges[n.id]]
    child_lps = [_logpdf_ad(c, data, row, θ, pm) for c in children(n)]
    return _logsumexp(logits .+ child_lps) - _logsumexp(logits)
end

function _logpdf_ad(n::ProductNode, data::EncodedData, row::Int, θ::AbstractVector, pm::ParamMap)
    s = zero(eltype(θ))
    for c in children(n)
        s += _logpdf_ad(c, data, row, θ, pm)
    end
    return s
end

function _logpdf_ad(n::Leaf, data::EncodedData, row::Int, θ::AbstractVector, pm::ParamMap)
    x = _get(data, row, n.scope)
    ismissing(x) && return zero(eltype(θ))
    kind = pm.leaf_kind[n.id]
    ϕ = @view θ[pm.ranges[n.id]]
    return _leaf_logpdf(kind, ϕ, x)
end

function _leaf_logpdf(kind::Symbol, ϕ, x)
    if x isa AbstractInterval
        return _leaf_interval_logpdf(kind, ϕ, x)
    elseif _is_set_observation(x)
        return _leaf_set_logpdf(kind, ϕ, x)
    end
    return _leaf_point_logpdf(kind, ϕ, x)
end

function _leaf_point_logpdf(kind::Symbol, ϕ, x)
    if kind === :normal
        μ = ϕ[1]
        σ = _softplus(ϕ[2]) + _POS_EPS
        z = (x - μ) / σ
        return -log(σ) - 0.5 * (log(2π) + z^2)
    elseif kind === :gamma
        α = _softplus(ϕ[1]) + _POS_EPS
        θ = _softplus(ϕ[2]) + _POS_EPS
        x < 0 && return -Inf
        if x == 0
            # Avoid NaNs from (α-1)*log(0) when α≈1. Handle boundary explicitly.
            if α < 1
                return Inf
            elseif abs(α - 1) < 1e-8
                return -log(θ)
            else
                return -Inf
            end
        end
        return (α - 1) * log(x) - x / θ - α * log(θ) - SpecialFunctions.loggamma(α)
    elseif kind === :poisson
        λ = _softplus(ϕ[1]) + _POS_EPS
        !_is_integer_observation(x) && return -Inf
        k = Int(x)
        k < 0 && return -Inf
        return k * log(λ) - λ - SpecialFunctions.logfactorial(k)
    elseif kind === :negbin
        r = _softplus(ϕ[1]) + _POS_EPS
        p = _safe_prob(ϕ[2])
        !_is_integer_observation(x) && return -Inf
        k = Int(x)
        k < 0 && return -Inf
        return SpecialFunctions.loggamma(k + r) - SpecialFunctions.loggamma(r) - SpecialFunctions.logfactorial(k) + r * log(p) + k * log1p(-p)
    elseif kind === :categorical
        idx = Int(x)
        logits = ϕ
        1 <= idx <= length(logits) || return -Inf
        return logits[idx] - _logsumexp(logits)
    end
    error("Unsupported leaf kind: $kind")
end

function _leaf_interval_logpdf(kind::Symbol, ϕ, x::AbstractInterval)
    isempty(x) && return -Inf
    if kind === :normal
        μ = ϕ[1]
        σ = _softplus(ϕ[2]) + _POS_EPS
        lo, hi = leftendpoint(x), rightendpoint(x)
        return _logprobdiff(_normal_cdf(hi, μ, σ), _normal_cdf(lo, μ, σ))
    elseif kind === :gamma
        α = _softplus(ϕ[1]) + _POS_EPS
        θ = _softplus(ϕ[2]) + _POS_EPS
        lo, hi = leftendpoint(x), rightendpoint(x)
        return _logprobdiff(_gamma_cdf(hi, α, θ), _gamma_cdf(lo, α, θ))
    elseif kind === :poisson || kind === :negbin || kind === :categorical
        return _leaf_set_logpdf(kind, ϕ, _integer_values(x))
    end
    error("Interval observations are unsupported for leaf kind: $kind")
end

function _leaf_set_logpdf(kind::Symbol, ϕ, xs)
    isempty(xs) && return -Inf
    if kind === :categorical
        idxs = [Int(x) for x in xs if 1 <= Int(x) <= length(ϕ)]
        isempty(idxs) && return -Inf
        return _logsumexp(ϕ[idxs]) - _logsumexp(ϕ)
    elseif kind === :poisson || kind === :negbin
        return _logsumexp([_leaf_point_logpdf(kind, ϕ, x) for x in xs])
    end
    error("Set-valued observations are only supported for discrete leaves; got $kind")
end

_is_integer_observation(x) = x isa Integer || (x isa Real && isinteger(x))

function _integer_values(x::AbstractInterval)
    lo, hi = leftendpoint(x), rightendpoint(x)
    (!isfinite(lo) || !isfinite(hi)) && error("Discrete interval observations must have finite endpoints")
    first = isleftclosed(x) ? ceil(Int, lo) : floor(Int, lo) + 1
    last = isrightclosed(x) ? floor(Int, hi) : ceil(Int, hi) - 1
    return first:last
end

function _logprobdiff(hi, lo)
    p = hi - lo
    return p > zero(p) ? log(p) : -Inf
end

function _normal_cdf(x, μ, σ)
    x == -Inf && return zero(μ + σ)
    x == Inf && return one(μ + σ)
    z = (x - μ) / σ
    return 0.5 * SpecialFunctions.erfc(-z / sqrt(2))
end

function _gamma_cdf(x, α, θ)
    x <= 0 && return zero(α + θ)
    x == Inf && return one(α + θ)
    return first(SpecialFunctions.gamma_inc(α, x / θ))
end

function with_params(spn::SumProductNetwork, θ::AbstractVector, pm::ParamMap)
    root = _with_params(spn.root, θ, pm)
    return SumProductNetwork(root, spn.categorical_pool, spn.ScM)
end

function _with_params(n::SumNode, θ::AbstractVector, pm::ParamMap)
    logits = @view θ[pm.ranges[n.id]]
    logw = collect(logits .- LogExpFunctions.logsumexp(logits))
    kids = Node[_with_params(c, θ, pm) for c in children(n)]
    return SumNode(logw, kids, n.scope, n.id)
end

function _with_params(n::ProductNode, θ::AbstractVector, pm::ParamMap)
    kids = Node[_with_params(c, θ, pm) for c in children(n)]
    return ProductNode(kids, n.scope, n.id)
end

function _with_params(n::Leaf, θ::AbstractVector, pm::ParamMap)
    kind = pm.leaf_kind[n.id]
    ϕ = @view θ[pm.ranges[n.id]]
    dist = _leaf_dist(kind, ϕ)
    return Leaf(dist, n.scope, n.id)
end

function _leaf_dist(kind::Symbol, ϕ)
    if kind === :normal
        return Normal(ϕ[1], _softplus(ϕ[2]) + _POS_EPS)
    elseif kind === :gamma
        return Gamma(_softplus(ϕ[1]) + _POS_EPS, _softplus(ϕ[2]) + _POS_EPS)
    elseif kind === :poisson
        return Poisson(_softplus(ϕ[1]) + _POS_EPS)
    elseif kind === :negbin
        return NegativeBinomial(_softplus(ϕ[1]) + _POS_EPS, _safe_prob(ϕ[2]))
    elseif kind === :categorical
        logits = ϕ
        logZ = LogExpFunctions.logsumexp(logits)
        ps = exp.(logits .- logZ)
        return Categorical(ps)
    end
    error("Unsupported leaf kind: $kind")
end

function fit_params(
    spn::SumProductNetwork,
    X;
    θ0 = nothing,
    pm::Union{Nothing,ParamMap} = nothing,
    maxiters::Int = 200,
    lr::Real = 1e-2,
    encoded::Bool = false,
    verbose::Bool = true,
)
    if θ0 === nothing || pm === nothing
        θ0′, pm′ = initial_params(spn)
        θ0 === nothing && (θ0 = θ0′)
        pm === nothing && (pm = pm′)
    end

    data = encoded ? X : encode_data(spn, X)
    loss(θ) = -meanlogpdf(spn, data, θ, pm; encoded = true)

    opt = Optimisers.Adam(lr)
    st = Optimisers.setup(opt, θ0)
    θ = copy(θ0)
    history = Float64[]

    for it in 1:maxiters
        l, back = Zygote.pullback(loss, θ)
        g = first(back(one(l)))
        st, θ = Optimisers.update(st, θ, g)
        push!(history, l)
        if verbose && (it == 1 || it % 25 == 0 || it == maxiters)
            @info "fit_params" iter = it loss = l
        end
    end

    return θ, pm, history
end

function fit_parameters(spn::SumProductNetwork, X; kwargs...)
    θ, pm, history = fit_params(spn, X; kwargs...)
    return with_params(spn, θ, pm), θ, pm, history
end
