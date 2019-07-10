# LearnSPN
struct ℵ
    init::Float64
    temp::Float64
end

export factor_v, learnSPN


function learnSPN(X, α_init=0.1)
    α = ℵ(α_init, α_init) # ("global" alpha, "local" alpha)
    init = SumNode() # Can always Init with a sum node. Weight = 1 by default so just add below it.
    SPN = learnSPN!(init, X, collect(1:size(X, 2)), BitVector(fill(true, size(X, 1))), weight = 1, α = α)[1] # Take child as root, ignore init.
    cat_vars = Dict(i=>col[findfirst(x->!ismissing(x), col)].pool for (i,col) in enumerate(columns(X)) if eltype(col) <: Union{Missing,CategoricalString})
    SPN = SumProductNetwork(SPN, cat_vars)
    return SPN
end

# Change to mutating. Modify calls
function learnSPN!(Node, X, scope, obs; weight = 1, α = ℵ(0.1, 0.1))
    if length(scope) == 1 # V is a univariate
        add_univariate_leaf!(Node, X, weight, scope, obs)
    else
        factored_scopes = factor(X[obs], scope, α.temp, kind=:HSIC) # Need to pass entire X and current scope. Also, no need to attempt to factor twice in a row.
        if length(factored_scopes) ≥ 2 # 1 if data doesn't factor.
            Child = ProductNode()
            for scope in factored_scopes
                learnSPN!(Child, X, scope, obs, weight = 1, α = ℵ(α.init, α.init))
            end
            Node isa SumNode ? add!(Node, Child, log(weight)) : add!(Node, Child)
        else
            clustered_obs = cluster(select(X, Tuple(scope)), obs)
            if length(clustered_obs) ≥ 2 #clustering_works and has enough present obs.
                Child = SumNode()
                for ob in clustered_obs
                    learnSPN!(Child, X, scope, ob, weight = sum(ob)/sum(obs), α = ℵ(α.init, α.init))
                end
                Node isa SumNode ? add!(Node, Child, log(weight)) : add!(Node, Child)
            else
                #Node = add_multivariate_leaf(Node, Class, X, weight, scope)
                learnSPN!(Node, X, scope, obs, weight = weight, α = ℵ(α.init, α.temp/2)) # Step halfway to zero
            end
        end
    end
    return(Node)
end

export LearnRSPN

function add_univariate_leaf!(SPN, Data_mat, weight, scope, obs)
    x = collect(skipmissing(select(Data_mat, scope[1])[obs]))
    D = fit_dist(x)
    SPN isa SumNode ? add!(SPN, Leaf(D, scope[1]), log(weight)) : add!(SPN, Leaf(D, scope[1])) # Still need to do this.
    nothing
end

export add_univariate_leaf!

function fit_dist(x::AbstractVector{<:CategoricalString})
    D = fit(Categorical, [el.level for el in x])
    return D
end

function fit_dist(x::AbstractVector{<:Real})
    x̄, σ² = mean(x), var(x)
    if all(isinteger.(x))
        if σ² ≤ x̄ # Let's do Poisson
            D = Poisson(x̄)
        else
            p = x̄/σ²
            r = x̄^2/(σ²-x̄)
        #D = MixtureModel([C₁, NegativeBinomial(r,p)], [p₀, 1-p₀]) # Mixture, even though support
            D = NegativeBinomial(r,p)
        end
    else # Floats
        if all(x .≥ 0.)
            D = fit(Gamma, x)
        else
            D = fit(Normal, x)
        end
    end
    return D
end

function factor(X, scope, α = 0.1; kind = :HSIC, nmax_force_factor = 10) # X is the n x V dataset.
    # For HSIC, α is the probability of incorrectly rejecting the Null (that cols are indep) given the Null.
    # So lower corresponds to a higher chance of declaring independent.
    D = test_similarity(X, scope, α = α, kind = kind) # D, the dependency matrix.
    # Need to force dependency between same-part features.
    conns = [scope[con] for con in connected_components(Graph(D))]
    # This stuff is fallback factorization so we don't recurse forever
    if (length(conns)==1) && ((size(X, 1) ≤ nmax_force_factor)) # Factored a Part out.
        push!(conns, [pop!(conns[1])])
    end
    return(conns)
end
export factor

convert_missing(ar::Matrix{Union{Missing,T}}) where T<:Union{Real,CategoricalString} = convert(Matrix{T}, ar)
convert_missing(ar::Vector{Union{Missing,T}}) where T<:Union{Real,CategoricalString} = convert(Vector{T}, ar)
convert_missing(ar) = ar
#
# get_scales(X) = map(col -> sqrt(var(col)), eachcol(X))
#
# import Distributions: scale,scale!
# function scale!(X, σs)
#     for i in 1:size(X, 2)
#         X[:, i] .= X[:, i]./σs[i]
#     end
# end

# scale(X, σs) = reduce(hcat, [X[:, i]/σs[i] for i in 1:size(X, 2)])

function cluster(X, obs, min_obs = 5)
    df = X[obs]
    any_nonmiss = mapreduce(x->.!ismissing.(x), (x,y)->x .| y, columns(df))
    assignments = rcopy(R"cluster::pam(as.matrix(cluster::daisy(x = $(DataFrames.DataFrame(df))))[$any_nonmiss,$any_nonmiss], 2)$clustering")
    left_instances = BitVector(undef, size(df, 1))
    left_instances[any_nonmiss] .= (assignments .== 1)
    left_instances[.!any_nonmiss] .= rand(Bool, sum(.!any_nonmiss))
    for col in columns(X[obs])
        nonmissings = .!ismissing.(col)
        n_l = sum(left_instances .& nonmissings)
        n_r = sum(.!left_instances .& nonmissings)
        if (n_l < min_obs) || (n_r < min_obs)
            # Can add logic to "validify" the clusters by assigning "closest" observations that have the relevant attribuet.
            return(())
        end
    end
    l_obs,r_obs = copy(obs),copy(obs)
    l_obs[obs] .= left_instances
    r_obs[obs] .= .!left_instances
    return(l_obs, r_obs)
end

# function cluster(X, min_obs = 5)
#     # Clustering will find clusters based on complete caases and assign Ts based on closeness in their non-NaN components.
#     nonmissing_X = .!ismissing.(X.array)
#     complete_cases = reduce(&, nonmissing_X, dims = 2)[:]
#     X_copy = convert_missing(X[complete_cases, :].array)
#     σs = get_scales(X_copy)
#     scale!(X_copy, σs)
#     kfits = kmeans(X_copy', 2) # Hardcode two clusters for now. Can do cross validation for k later.
#     #if kfits.assignments
#     # Change assignment map to
#     c₁, c₂ = kfits.centers[:, 1], kfits.centers[:, 2]
#     left_instances = [dist_non_missing(v,c₁) < dist_non_missing(v,c₂) ? true : false for v in eachrow(scale(X.array, σs))]
#     for col in eachcol(nonmissing_X)
#         n_l = sum(col .& left_instances)
#         n_r = sum(col) - n_l
#         if (n_l < min_obs) || (n_r < min_obs)
#             # Can add logic to "validify" the clusters by assigning "closest" observations that have the relevant attribuet.
#             return(())
#         end
#     end
#     return(left_instances, .!left_instances)
# end

export cluster,indeptest

function indeptest(X::Vector{<:Number},Y::Vector{<:Number}, α)
    value, threshold = gammaHSIC(X, Y, α = α, randomSubSet = 1000)
    return ifelse((value < threshold) | isnan(value) | isnan(threshold), 0, 1)
end

function indeptest(X::Vector{<:Number},Y::CategoricalArray, α)
    # Just doing a Kruskal Wallis test
    # Need to split X into groups defined by Y and call KruskallWallisTest()
    if length(Y.pool)==1
        return(0)
    else
        groups = splitby(X, Y)
        p = pvalue(KruskalWallisTest(groups...))
        return ifelse(p < α, 1, 0)
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

indeptest(X::CategoricalArray,Y::Vector{<:Number}, α) = indeptest(Y,X, α)

function indeptest(X::CategoricalArray,Y::CategoricalArray, α)
    if length(Y.pool)==1 || length(X.pool)==1
        return(0)
    else
        d = counts(X.refs, Y.refs)
        p = pvalue(PowerDivergenceTest(d, lambda = 1.))
        return ifelse(p < α, 1, 0)
    end
end

# Modify to find "approximate independence", otherwise will always reject hypothesis with enough data.
# Right now, this problem is addressed by limiting # of obs for HSIC. Also speeds up calculation.
function test_similarity(X, scope = 1:size(X, 2); α = 0.05, kind = :HSIC) # α for test with H_0: independence.
    @assert kind in [:HSIC] "only supports HSIC right now since that tests H₀: Indep."
    Dep = NamedArray(Array{Int64,2}(undef,(size(X, 2), size(X, 2))), (collect(colnames(X)), collect(colnames(X)))) # 1 means dependent.
    Dep[diagind(Dep)] .= 1
    for colpair in combinations(scope,2)
        i,j = colpair[1],colpair[2]
        nonmissing_inds = .!ismissing.(select(X, i)) .& .!ismissing.(select(X, j))
        if sum(nonmissing_inds) == 0
            Dep[i,j] = Dep[j,i] = 0
        else
            samps = sum(nonmissing_inds) > 1000 ? sample(1:sum(nonmissing_inds), 1000) : 1:sum(nonmissing_inds)
            Dep[i,j] = Dep[j,i] = indeptest(convert_missing(select(X, i)[nonmissing_inds][samps]), convert_missing(select(X, j)[nonmissing_inds][samps]), α)
            # elseif kind==:kendall
            #     r = corkendall(X.array[nonmissing_inds, i], X.array[nonmissing_inds, j])
            #     n = sum(nonmissing_inds)
            #     t = r/sqrt(2(2n+5)/(9n*(n-1)))
            #     t₀ = quantile(TDist(n-2), α/2), quantile(TDist(n-2), 1 - α/2) # Outside this range is H_1: dependent
            #     Dep[i,j] = Dep[j,i] = ifelse(t₀[1] < t ≤ t₀[2], 0, 1)
            # elseif kind==:pearson
            #     r = cor(X.array[nonmissing_inds, i], X.array[nonmissing_inds, j])
            #     n = sum(nonmissing_inds)
            #     t = r*sqrt(n-2)/sqrt(1 - r^2)
            #     t₀ = quantile(Normal(), α/2), quantile(Normal(), 1 - α/2) # Outside this range is H_1: dependent
            #     Dep[i,j] = Dep[j,i] = ifelse(t₀[1] < t ≤ t₀[2], 0, 1) # Outside this range is H_1: dependent
        end
    end
    return(Dep[scope, scope])
end

export get_data_matrix, test_similarity

# Note:
# I have not implemented a way to allow for attributes to be binary/categorical. Worth doing after refactoring the code
