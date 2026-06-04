# LearnSPN
struct ℵ
    init::Float64
    temp::Float64
end

export learnSPN


function learnSPN(X, α_init=0.1)
    _validate_learnSPN_data(X)
    α = ℵ(α_init, α_init) # ("global" alpha, "local" alpha)
    init = SumNode() # Can always Init with a sum node. Weight = 1 by default so just add below it.
    scopemap = ScopeMap(columnnames(X), nonmissingtype.(eltype.(values(columns(X)))))
    SPN = learnSPN!(init, X, scopemap, weight = 1, α = α)[1] # Take child as root, ignore init.
    cat_vars = Dict{Int64,Any}(i=>col.pool for (i,col) in enumerate(columns(X)) if col isa AbstractCategoricalVector)
    SPN = SumProductNetwork(SPN, cat_vars, scopemap)
    return SPN
end

# Change to mutating. Modify calls
function learnSPN!(n, X, ScM; weight = 1, α = ℵ(0.1, 0.1))
    if dims(X, 2) == 1 # V is a univariate
        add_univariate_leaf!(n, X, ScM, weight)
    else
        V = factor(X, α.temp) # Need to pass entire X and current scope. Also, no need to attempt to factor twice in a row.
        if length(V) ≥ 2 # 1 if data doesn't factor.
            child = ProductNode()
            for Vᵢ in V
                learnSPN!(child, select(X, Vᵢ), ScM, weight = 1, α = ℵ(α.init, α.init))
            end
            n isa SumNode ? add!(n, child, log(weight)) : add!(n, child)
        else
            T = cluster(X)
            if length(T) ≥ 2 #clustering_works and has enough present obs.
                child = SumNode()
                for Tⱼ in T
                    learnSPN!(child, X[Tⱼ], ScM, weight = sum(Tⱼ)/length(Tⱼ), α = ℵ(α.init, α.init))
                end
                n isa SumNode ? add!(n, child, log(weight)) : add!(n, child)
            else
                #n = add_multivariate_leaf(n, Class, X, weight, scope)
                learnSPN!(n, X, ScM, weight = weight, α = ℵ(α.init, NaN)) # Force factorization
            end
        end
    end
    return(n)
end

function add_univariate_leaf!(SPN, X, ScM, weight)
    scope = ScM[columnnames(X)[1]]
    x = select(X,1)
    D = fit_dist(x)
    SPN isa SumNode ? add!(SPN, Leaf(D, scope), log(weight)) : add!(SPN, Leaf(D, scope)) # Still need to do this.
    nothing
end

export add_univariate_leaf!
export fit_dist

function _validate_learnSPN_data(X)
    for (nm, col) in zip(columnnames(X), columns(X))
        _validate_learnSPN_column(nm, col)
    end
    return nothing
end

function _validate_learnSPN_column(name, col)
    any(x -> !ismissing(x), col) ||
        throw(ArgumentError("Column $(name) has no observed values; cannot learn an SPN leaf distribution."))
    T = nonmissingtype(eltype(col))
    (T <: Number || col isa AbstractCategoricalVector) ||
        throw(ArgumentError("Column $(name) has unsupported eltype $(eltype(col)); expected numeric or categorical data."))
    return nothing
end

function _observed_values(x)
    vals = collect(skipmissing(x))
    isempty(vals) && throw(ArgumentError("Cannot fit a distribution to a column with no observed values."))
    return vals
end

function fit_dist(x::AbstractCategoricalVector)
    vals = _observed_values(x)
    D = fit(Categorical, [el.ref for el in vals])
    return D
end

function fit_dist(x::AbstractVector)
    vals = _observed_values(x)
    if all(isinteger.(vals))
        x̄, σ² = mean(vals), var(vals)
        if σ² ≤ x̄ # Let's do Poisson
            D = Poisson(x̄)
        else
            p = x̄/σ²
            r = x̄^2/(σ²-x̄)
        #D = MixtureModel([C₁, NegativeBinomial(r,p)], [p₀, 1-p₀]) # Mixture, even though support
            D = NegativeBinomial(r,p)
        end
    else # Floats
        if all(vals .> 0.)
            D = fit(Gamma, vals)
        else
            D = fit(Normal, vals)
        end
    end
    return D
end

function factor(X, α = 0.1) # X is the n x V dataset.
    # For HSIC, α is the probability of incorrectly rejecting the Null (that cols are indep) given the Null.
    # So lower corresponds to a higher chance of declaring independent.
    P = test_similarity(X) # D, the dependency matrix.
    Dep = similar(P, Float64)
    Dep[diagind(Dep)] .= 1
    pvals = sort(P.array[:])
    if isnan(α) # force factorization
        while isnan(α)
            a = pop!(pvals)
            Dep .= ifelse.(P .< a, 1, 0)
            conns = connected_components(Graph(Dep))
            if length(conns) > 1
                break
            end
        end
    else
        Dep .= ifelse.(P .< α, 1, 0)
        conns = connected_components(Graph(Dep))
    end
    return(conns)
end
export factor

convert_missing(ar::Matrix{Union{Missing,T}}) where {T<:Real} = convert(Matrix{T}, ar)
convert_missing(ar::Vector{Union{Missing,T}}) where {T<:Real} = convert(Vector{T}, ar)
convert_missing(ar::AbstractCategoricalVector{Union{Missing,T}}) where {T<:String} = convert(CategoricalVector{T}, ar)
convert_missing(ar) = ar

function replace_missings(D::Table)
    ready = _cluster_ready_table(D)
    ready === nothing && throw(ArgumentError("Cannot prepare clustering data with all-missing columns."))
    return ready
end

function _cluster_ready_table(D::Table)
    dat = []
    for (nm, col) in zip(columnnames(D), columns(D))
        ready_col = _cluster_ready_column(col, nm)
        ready_col === nothing && return nothing
        push!(dat, ready_col)
    end
    return Table(NamedTuple{columnnames(D)}(dat))
end

function _cluster_ready_column(col::AbstractCategoricalVector, name)
    refs = [el.ref for el in skipmissing(col)]
    isempty(refs) && return nothing
    Missing <: eltype(col) || return col
    mode_ref = _modal_ref(refs, length(levels(col)))
    out = copy(col)
    out[ismissing.(out)] .= levels(col)[mode_ref]
    return out
end

function _cluster_ready_column(col, name)
    T = nonmissingtype(eltype(col))
    if T <: Number
        vals = collect(skipmissing(col))
        isempty(vals) && return nothing
        μ = mean(vals)
        return [ismissing(x) ? Float64(μ) : Float64(x) for x in col]
    end
    throw(ArgumentError("Column $(name) has unsupported eltype $(eltype(col)); expected numeric or categorical data."))
end

function _modal_ref(refs, nlevels::Int)
    level_counts = counts(refs, 1:nlevels)
    return argmax(level_counts)
end

function cluster(X, min_obs = 5)
    D = _cluster_ready_table(X)
    D === nothing && return (())
    U,Δ,_ = pcamix(D)
    coords = U*Δ
    if size(coords, 2) < 1 || size(coords, 1) < 3 # Categorical, all same level, or n <= k
        return (())
    end
    assignments = kmeans(coords', 2).assignments
    left_instances = (assignments .== 1)
    for col in columns(X)
        nonmissings = .!ismissing.(col)
        n_l = sum(left_instances .& nonmissings)
        n_r = sum(.!left_instances .& nonmissings)
        if (n_l < min_obs) || (n_r < min_obs)
            # Can add logic to "validify" the clusters by assigning "closest" observations that have the relevant attribuet.
            return (())
        end
    end
    return(left_instances, .!left_instances)
end

export cluster,indeptest

_observed_refs(Y::CategoricalArray) = unique(filter(!=(0), Y.refs))

function indeptest(X::Vector{<:Number},Y::Vector{<:Number})
    X,Y = convert(Vector{Float64},X),convert(Vector{Float64},Y)
    p = gammaHSIC(X, Y)
    return ifelse(isnan(p), 1., p)
end

function indeptest(X::Vector{<:Number},Y::CategoricalArray)
    # Just doing a Kruskal Wallis test
    # Need to split X into groups defined by Y and call KruskallWallisTest()
    if length(_observed_refs(Y)) <= 1
        return(1.)
    else
        kwtest = @suppress begin
            KruskalWallisTest(splitby(X, Y)...)
        end
        p = pvalue(kwtest)
        return ifelse(isnan(p), 1., p)
    end
end

export splitby
function splitby(X::Vector{<:Number}, F::CategoricalArray)
    refs = sort(_observed_refs(F))
    ref_index = Dict(ref => i for (i, ref) in enumerate(refs))
    split_X = [typeof(X)() for _ in refs]
    for (x,f) in zip(X,F.refs)
        f == 0 && continue
        push!(split_X[ref_index[f]], x)
    end
    return(split_X)
end

indeptest(X::CategoricalArray,Y::Vector{<:Number}) = indeptest(Y,X)

function indeptest(X::CategoricalArray,Y::CategoricalArray)
    nonmissing = (X.refs .!= 0) .& (Y.refs .!= 0)
    sum(nonmissing) < 2 && return 1.
    xrefs, yrefs = X.refs[nonmissing], Y.refs[nonmissing]
    (length(unique(xrefs)) <= 1 || length(unique(yrefs)) <= 1) && return 1.
    d = counts(xrefs, yrefs)
    if any(size(d).==1)
        return(1.)
    end
    return pvalue(PowerDivergenceTest(d, lambda = 1.))
end

# Modify to find "approximate independence", otherwise will always reject hypothesis with enough data.
# Right now, this problem is addressed by limiting # of obs for indep tests.
function test_similarity(X; ntest_samps = 100, min_pairwise_obs = 2) # p val for test with H_0: independence.
    P = NamedArray(Array{Float64,2}(undef,(dims(X, 2), dims(X, 2))), (collect(columnnames(X)), collect(columnnames(X)))) # 1 means dependent.
    P[diagind(P)] .= 0.
    for colpair in combinations(columnnames(X), 2)
        i,j = colpair[1],colpair[2]
        nonmissing_inds = .!ismissing.(select(X, i)) .& .!ismissing.(select(X, j))
        n_nonmissing = sum(nonmissing_inds)
        if n_nonmissing < min_pairwise_obs
            P[i,j] = P[j,i] = 1. # i.e. assume Independent
        else
            samps = n_nonmissing > ntest_samps ? sample(1:n_nonmissing, ntest_samps) : 1:n_nonmissing
            P[i,j] = P[j,i] = indeptest(convert_missing(select(X, i)[nonmissing_inds][samps]), convert_missing(select(X, j)[nonmissing_inds][samps]))
        end
    end
    return(P)
end

export get_data_matrix, test_similarity
