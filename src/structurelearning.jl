# LearnSPN
struct ℵ
    init::Float64
    temp::Float64
end

export factor_v, learnSPN


function learnSPN(X, α_init=0.1)
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

export LearnRSPN

function add_univariate_leaf!(SPN, X, ScM, weight)
    scope = ScM[columnnames(X)[1]]
    x = select(X,1)
    D = fit_dist(x)
    SPN isa SumNode ? add!(SPN, Leaf(D, scope), log(weight)) : add!(SPN, Leaf(D, scope)) # Still need to do this.
    nothing
end

export add_univariate_leaf!
export fit_dist
function fit_dist(x::AbstractCategoricalVector)
    D = Missing <: eltype(x) ? fit(Categorical, [el.level for el in skipmissing(x)]) : fit(Categorical, [el.level for el in x])
    return D
end

function fit_dist(x::AbstractVector)
    if all(isinteger.(skipmissing(x)))
        x̄, σ² = Missing <: eltype(x) ? (mean(skipmissing(x)), var(skipmissing(x))) : (mean(x), var(x))
        if σ² ≤ x̄ # Let's do Poisson
            D = Poisson(x̄)
        else
            p = x̄/σ²
            r = x̄^2/(σ²-x̄)
        #D = MixtureModel([C₁, NegativeBinomial(r,p)], [p₀, 1-p₀]) # Mixture, even though support
            D = NegativeBinomial(r,p)
        end
    else # Floats
        if all(skipmissing(x) .> 0.)
            D = Missing <: eltype(x) ? fit(Gamma, collect(skipmissing(x))) : fit(Gamma, x)
        else
            D = Missing <: eltype(x) ? fit(Normal, collect(skipmissing(x))) : fit(Normal, x)
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

# Finish this....
function replace_missings(D::Table)
    dat = []
    for (nm,col) in zip(columnnames(D), columns(D))
        if Missing <: eltype(col)
            if nonmissingtype(eltype(col)) <: Number
                col = col .- mean(skipmissing(col))
                col[ismissing.(col)] .= 0
                col = convert(Vector{nonmissingtype(eltype(col))}, col)
            elseif nonmissingtype(eltype(col)) <: CategoricalString
                "?" in levels(col) ? error("Attempting to recode categorical missing to value: \"?\"") : nothing
                col = recode(col, missing=>"?")
            else
                error("Features must have Number or CategoricalString eltypes (Missings allowed).")
            end
        end
        push!(dat, col)
    end
    return Table(NamedTuple{columnnames(D)}(dat))
end

function cluster(X, min_obs = 5)
    # Need to skipmissing or find some way to handle missings.
    # Instead, just replace missing with the column mean and cast it as another level for the categoricals.
    D = replace_missings(X)
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

function indeptest(X::Vector{<:Number},Y::Vector{<:Number})
    X,Y = convert(Vector{Float64},X),convert(Vector{Float64},Y)
    p = gammaHSIC(X, Y)
    return ifelse(isnan(p), 1., p)
end

function indeptest(X::Vector{<:Number},Y::CategoricalArray)
    # Just doing a Kruskal Wallis test
    # Need to split X into groups defined by Y and call KruskallWallisTest()
    if length(Y.pool)==1
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
    split_X = [typeof(X)() for _ in 1:length(F.pool)]
    for (x,f) in zip(X,F.refs)
        push!(split_X[f], x)
    end
    return(split_X)
end

indeptest(X::CategoricalArray,Y::Vector{<:Number}) = indeptest(Y,X)

function indeptest(X::CategoricalArray,Y::CategoricalArray)
    d = counts(X.refs, Y.refs)
    if any(size(d).==1)
        return(1.)
    end
    return pvalue(PowerDivergenceTest(d, lambda = 1.))
end

# Modify to find "approximate independence", otherwise will always reject hypothesis with enough data.
# Right now, this problem is addressed by limiting # of obs for indep tests.
function test_similarity(X; ntest_samps = 100) # p val for test with H_0: independence.
    P = NamedArray(Array{Float64,2}(undef,(dims(X, 2), dims(X, 2))), (collect(columnnames(X)), collect(columnnames(X)))) # 1 means dependent.
    P[diagind(P)] .= 0.
    for colpair in combinations(columnnames(X), 2)
        i,j = colpair[1],colpair[2]
        nonmissing_inds = .!ismissing.(select(X, i)) .& .!ismissing.(select(X, j))
        if sum(nonmissing_inds) == 0
            P[i,j] = P[j,i] = 1. # i.e. assume Independent
        else
            samps = sum(nonmissing_inds) > ntest_samps ? sample(1:sum(nonmissing_inds), ntest_samps) : 1:sum(nonmissing_inds)
            P[i,j] = P[j,i] = indeptest(convert_missing(select(X, i)[nonmissing_inds][samps]), convert_missing(select(X, j)[nonmissing_inds][samps]))
        end
    end
    return(P)
end

export get_data_matrix, test_similarity
