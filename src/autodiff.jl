export ParamMap, EncodedData, NonFiniteTrainingError
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

struct NonFiniteTrainingError <: Exception
    stage::Symbol
    iter::Int
    value
    index::Union{Nothing,Int}
    node_id::Union{Nothing,UInt128}
    node_kind::Union{Nothing,Symbol}
    range::Union{Nothing,UnitRange{Int}}
end

function Base.showerror(io::IO, err::NonFiniteTrainingError)
    print(io, "Nonfinite autodiff training value during ", err.stage)
    err.iter > 0 && print(io, " at iteration ", err.iter)
    print(io, ": ", err.value)
    if err.index !== nothing
        print(io, " at parameter index ", err.index)
    end
    if err.node_id !== nothing
        print(io, " for ", err.node_kind, " node ", err.node_id)
    end
    if err.range !== nothing
        print(io, " with parameter range ", err.range)
    end
end

@inline _get(data::EncodedData, row::Int, col::Int) = data.cols[col][row]

function validate_training_data(spn::SumProductNetwork, X, pm::ParamMap; encoded::Bool = false)
    _validate_param_map(spn, pm)
    schema = _training_schema(spn, X; encoded = encoded)
    kinds = _leaf_kinds_by_scope(spn, pm)
    for j in 1:schema.ncols
        col_kinds = get(kinds, j, Symbol[])
        isempty(col_kinds) && throw(ArgumentError("No leaf parameters found for training column $j."))
        _validate_training_column(schema.cols[j], col_kinds, schema.categorical[j], schema.names[j])
    end
    return nothing
end

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

struct _TrainingSchema{C,N}
    cols::C
    names::N
    ncols::Int
    categorical::Vector{Bool}
end

function _training_schema(spn::SumProductNetwork, X::AbstractMatrix; encoded::Bool = false)
    n_expected = length(spn.ScM)
    size(X, 2) == n_expected || throw(ArgumentError("Training matrix has $(size(X, 2)) columns; expected $n_expected."))
    cols = [view(X, :, j) for j in 1:size(X, 2)]
    names = [string(spn.ScM[j]) for j in 1:n_expected]
    categorical = fill(false, n_expected)
    return _TrainingSchema(cols, names, n_expected, categorical)
end

function _training_schema(spn::SumProductNetwork, X::Table; encoded::Bool = false)
    expected = Tuple(keys(spn.ScM))
    got = Tuple(columnnames(X))
    got == expected || throw(ArgumentError("Training table columns must match SPN scope order. Expected $(expected), got $(got)."))
    raw_cols = collect(columns(X))
    _validate_column_lengths(raw_cols)
    categorical = [haskey(spn.categorical_pool, j) for j in eachindex(raw_cols)]
    return _TrainingSchema(raw_cols, string.(collect(got)), length(raw_cols), categorical)
end

function _training_schema(spn::SumProductNetwork, X::EncodedData; encoded::Bool = false)
    n_expected = length(spn.ScM)
    length(X.cols) == n_expected || throw(ArgumentError("Encoded training data has $(length(X.cols)) columns; expected $n_expected."))
    all(length(col) == X.nrows for col in X.cols) || throw(ArgumentError("Encoded training columns must all have nrows=$(X.nrows)."))
    names = [string(spn.ScM[j]) for j in 1:n_expected]
    categorical = fill(false, n_expected)
    return _TrainingSchema(X.cols, names, n_expected, categorical)
end

function _validate_column_lengths(cols)
    isempty(cols) && return nothing
    n = length(cols[1])
    all(length(col) == n for col in cols) || throw(ArgumentError("Training table columns must all have the same length."))
    return nothing
end

function _leaf_kinds_by_scope(spn::SumProductNetwork, pm::ParamMap)
    kinds = Dict{Int,Vector{Symbol}}()
    function visit(n::Node)
        if n isa Leaf
            push!(get!(kinds, n.scope, Symbol[]), pm.leaf_kind[n.id])
        else
            for c in children(n)
                visit(c)
            end
        end
    end
    visit(spn.root)
    return Dict(k => unique(v) for (k, v) in kinds)
end

function _validate_param_map(spn::SumProductNetwork, pm::ParamMap, θ = nothing)
    expected = Dict{UInt128,Tuple{Symbol,Int}}()
    leaf_ids = Set{UInt128}()

    function visit(n::Node)
        if n isa SumNode
            expected[n.id] = (:sum, length(children(n)))
            foreach(visit, children(n))
        elseif n isa ProductNode
            foreach(visit, children(n))
        elseif n isa Leaf
            kind, ϕ0 = _leaf_unconstrained(spn, n)
            expected[n.id] = (kind, length(ϕ0))
            push!(leaf_ids, n.id)
        else
            error("Unsupported node type: $(typeof(n))")
        end
    end

    visit(spn.root)

    expected_ids = Set(keys(expected))
    for (id, (kind, nparams)) in expected
        haskey(pm.ranges, id) || throw(ArgumentError("ParamMap is missing a parameter range for $kind node $id."))
        length(pm.ranges[id]) == nparams ||
            throw(ArgumentError("ParamMap range for $kind node $id has length $(length(pm.ranges[id])); expected $nparams."))
        if kind === :sum
            !haskey(pm.leaf_kind, id) ||
                throw(ArgumentError("ParamMap leaf_kind has an entry for sum node $id; only leaves should have leaf kinds."))
        else
            haskey(pm.leaf_kind, id) || throw(ArgumentError("ParamMap is missing a leaf kind for $kind leaf node $id."))
            pm.leaf_kind[id] === kind ||
                throw(ArgumentError("ParamMap leaf kind for node $id is $(pm.leaf_kind[id]); expected $kind."))
        end
    end

    extra_ranges = setdiff(Set(keys(pm.ranges)), expected_ids)
    isempty(extra_ranges) ||
        throw(ArgumentError("ParamMap contains a range for unknown node $(first(extra_ranges))."))

    extra_leaf_kinds = setdiff(Set(keys(pm.leaf_kind)), leaf_ids)
    isempty(extra_leaf_kinds) ||
        throw(ArgumentError("ParamMap contains a leaf kind for unknown or non-leaf node $(first(extra_leaf_kinds))."))

    θ === nothing || _validate_param_ranges(pm, length(θ))
    return nothing
end

function _validate_param_ranges(pm::ParamMap, nparams::Int)
    seen = Set{Int}()
    for (id, r) in pm.ranges
        first(r) >= 1 || throw(ArgumentError("ParamMap range for node $id starts at $(first(r)); expected a positive index."))
        last(r) <= nparams ||
            throw(ArgumentError("ParamMap range for node $id ends at $(last(r)), but the parameter vector has length $nparams."))
        for idx in r
            idx in seen && throw(ArgumentError("ParamMap range for node $id overlaps another range at parameter index $idx."))
            push!(seen, idx)
        end
    end
    length(seen) == nparams ||
        throw(ArgumentError("Parameter vector has length $nparams, but ParamMap covers $(length(seen)) parameter indices."))
    return nothing
end

function _validate_training_column(col, kinds::Vector{Symbol}, iscategorical::Bool, name::AbstractString)
    for i in eachindex(col)
        _validate_observation(col[i], kinds, iscategorical, name, i)
    end
    return nothing
end

function _validate_observation(x, kinds::Vector{Symbol}, iscategorical::Bool, name::AbstractString, row)
    ismissing(x) && return nothing
    if x isa AbstractInterval
        iscategorical && throw(ArgumentError("Categorical interval observations are unsupported for column $(name), row $(row); use a finite set of category levels."))
        for kind in kinds
            if kind === :poisson || kind === :negbin
                _validate_discrete_interval(x, name, row)
            elseif !(kind === :normal || kind === :gamma)
                throw(ArgumentError("Interval observations are unsupported for $kind leaves in column $(name), row $(row)."))
            end
        end
    elseif _is_set_observation(x)
        isempty(x) && throw(ArgumentError("Set-valued observation for column $(name), row $(row) is empty."))
        for kind in kinds
            (kind === :poisson || kind === :negbin || kind === :categorical) ||
                throw(ArgumentError("Set-valued observations are unsupported for $kind leaves in column $(name), row $(row)."))
            _validate_set_observation(x, kind, iscategorical, name, row)
        end
    else
        for kind in kinds
            _validate_point_observation(x, kind, iscategorical, name, row)
        end
    end
    return nothing
end

function _validate_set_observation(xs, kind::Symbol, iscategorical::Bool, name, row)
    if kind === :poisson || kind === :negbin
        all(_is_integer_observation, xs) || throw(ArgumentError("Column $(name), row $(row) has a set-valued observation with non-integer values; $kind leaves require finite integer sets."))
    elseif kind === :categorical && !iscategorical
        all(x -> x isa Integer, xs) || throw(ArgumentError("Categorical column $(name), row $(row) must use encoded integer category ids for matrix/encoded set observations."))
    end
    return nothing
end

function _validate_discrete_interval(x::AbstractInterval, name, row)
    lo, hi = leftendpoint(x), rightendpoint(x)
    (lo isa Real && hi isa Real) ||
        throw(ArgumentError("Discrete interval observation for column $(name), row $(row) must have numeric endpoints."))
    return nothing
end

function _validate_point_observation(x, kind::Symbol, iscategorical::Bool, name, row)
    if kind === :normal || kind === :gamma
        x isa Real || throw(ArgumentError("Column $(name), row $(row) has $(typeof(x)); $kind leaves require numeric point observations."))
    elseif kind === :poisson || kind === :negbin
        _is_integer_observation(x) || throw(ArgumentError("Column $(name), row $(row) has $(x); $kind leaves require integer point observations, finite integer intervals, or finite sets."))
    elseif kind === :categorical
        if !iscategorical && !(x isa Integer)
            throw(ArgumentError("Categorical column $(name), row $(row) must use encoded integer category ids for matrix/encoded data."))
        end
    else
        throw(ArgumentError("Unsupported leaf kind $kind for column $(name)."))
    end
    return nothing
end

function _encode_categorical_observation(pool, v)
    ismissing(v) && return missing
    if v isa AbstractInterval
        throw(ArgumentError("Categorical interval observations are unsupported; use a finite set of category levels."))
    end
    if _is_set_observation(v)
        return [_encode_categorical_value(pool, el) for el in v]
    end
    return _encode_categorical_value(pool, v)
end

function _encode_categorical_value(pool, v)
    if v isa UInt32
        1 <= Int(v) <= length(pool.levels) || throw(ArgumentError("Encoded categorical value $(v) is outside 1:$(length(pool.levels))."))
        return v
    end
    key = v isa CategoricalValue ? String(v) : v isa AbstractString ? v : string(v)
    haskey(pool.invindex, key) || throw(ArgumentError("Unknown categorical level $(repr(key)); expected one of $(collect(pool.levels))."))
    return pool.invindex[key]
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
    data = if encoded
        ChainRulesCore.ignore_derivatives() do
            _validate_param_map(spn, pm, θ)
            validate_training_data(spn, X, pm; encoded = true)
        end
        X
    else
        ChainRulesCore.ignore_derivatives() do
            _validate_param_map(spn, pm, θ)
            validate_training_data(spn, X, pm)
            encode_data(spn, X)
        end
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
    elseif kind === :poisson || kind === :negbin
        return _leaf_discrete_interval_logpdf(kind, ϕ, x)
    elseif kind === :categorical
        return _leaf_set_logpdf(kind, ϕ, _integer_values(x))
    end
    error("Interval observations are unsupported for leaf kind: $kind")
end

function _leaf_discrete_interval_logpdf(kind::Symbol, ϕ, x::AbstractInterval)
    isempty(x) && return -Inf
    lower, upper = _integer_bounds(x)
    if upper !== nothing && upper < 0
        return -Inf
    end
    if lower !== nothing
        lower = max(lower, 0)
    end

    if lower === nothing && upper === nothing
        return zero(eltype(ϕ))
    elseif lower === nothing
        return _leaf_set_logpdf(kind, ϕ, 0:upper)
    elseif upper === nothing
        lower <= 0 && return zero(eltype(ϕ))
        excluded = _leaf_set_logpdf(kind, ϕ, 0:(lower - 1))
        return _log1mexp(excluded)
    else
        upper < lower && return -Inf
        return _leaf_set_logpdf(kind, ϕ, lower:upper)
    end
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
    first, last = _integer_bounds(x)
    (first === nothing || last === nothing) && error("Discrete interval observations must have finite endpoints")
    return first:last
end

function _integer_bounds(x::AbstractInterval)
    lo, hi = leftendpoint(x), rightendpoint(x)
    first = lo == -Inf ? nothing : isleftclosed(x) ? ceil(Int, lo) : floor(Int, lo) + 1
    last = hi == Inf ? nothing : isrightclosed(x) ? floor(Int, hi) : ceil(Int, hi) - 1
    return first, last
end

function _log1mexp(x)
    x == -Inf && return zero(x)
    x >= 0 && return -Inf
    return x < -log(2) ? log1p(-exp(x)) : log(-expm1(x))
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

function _assert_finite_scalar(x, pm::ParamMap, stage::Symbol, iter::Int)
    isfinite(x) && return nothing
    throw(NonFiniteTrainingError(stage, iter, x, nothing, nothing, nothing, nothing))
end

function _assert_finite_vector(xs, pm::ParamMap, stage::Symbol, iter::Int)
    idx = findfirst(x -> !isfinite(x), xs)
    idx === nothing && return nothing
    node_id, node_kind, range = _param_context(pm, idx)
    throw(NonFiniteTrainingError(stage, iter, xs[idx], idx, node_id, node_kind, range))
end

function _assert_gradient(g, pm::ParamMap, iter::Int)
    g === nothing && throw(NonFiniteTrainingError(:gradient, iter, nothing, nothing, nothing, nothing, nothing))
    return _assert_finite_vector(g, pm, :gradient, iter)
end

function _param_context(pm::ParamMap, idx::Int)
    for (node_id, range) in pm.ranges
        idx in range || continue
        return node_id, get(pm.leaf_kind, node_id, :sum), range
    end
    return nothing, nothing, nothing
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
    _validate_param_map(spn, pm, θ0)
    _assert_finite_vector(θ0, pm, :initial_parameters, 0)

    data = if encoded
        validate_training_data(spn, X, pm; encoded = true)
        X
    else
        validate_training_data(spn, X, pm)
        encode_data(spn, X)
    end
    loss(θ) = -meanlogpdf(spn, data, θ, pm; encoded = true)

    opt = Optimisers.Adam(lr)
    st = Optimisers.setup(opt, θ0)
    θ = copy(θ0)
    history = Float64[]

    for it in 1:maxiters
        l, back = Zygote.pullback(loss, θ)
        _assert_finite_scalar(l, pm, :loss, it)
        g = first(back(one(l)))
        _assert_gradient(g, pm, it)
        st, θ = Optimisers.update(st, θ, g)
        _assert_finite_vector(θ, pm, :updated_parameters, it)
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
