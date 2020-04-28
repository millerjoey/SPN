export rand

##### Unconditional Sampling #####
rand(SPN::SumProductNetwork, n::Integer = 1) = rand(SPN, n, fill(missing, length(SPN.ScM)))

#### Conditional Sampling #####

rand(SPN::SumProductNetwork, query::AbstractVector, n = 1) = rand(SPN::SumProductNetwork, n, query::AbstractVector)

# Allow query to be a dictionary. Dict("x1"=>5, "x2"=>1..2, "x5"=>"C") and assume the rest missing.
rand(SPN::SumProductNetwork, n::Integer, query::Dict) = rand(SPN, n, queryfromdict(SPN, query))

# Can possibly switch to use multithreading, making a single for loop.
function rand(SPN::SumProductNetwork, n::Integer, query::AbstractVector)
    # force query to contain static arrays maybe?
    query = deepcopy(query)
    for (scope,pool) in SPN.categorical_pool
        if !ismissing(query[scope])
            eltype(query) !== Any && (query = convert(Vector{Any}, query))
            if isa(query[scope], AbstractVector)
                query[scope] = [pool.invindex[v] for v in query[scope]]
            elseif isa(query[scope], String)
                query[scope] = pool.invindex[query[scope]]
            end
        end
    end
    samps = Dict{Int,Any}(1:length(SPN.ScM) .=> [Vector{ismissing(query[i]) ? T : typeof(query[i])}(undef, n) for (i,T) in enumerate(SPN.ScM.Types)])
    for i in eachindex(query)
        if !ismissing(query[i])
            samps[i] .= Ref(query[i])
        end
    end
    condsamp!(samps, SPN.root, 1:n, query)
    for (scope,pool) in SPN.categorical_pool
        samps[scope] = [pool.index[Int.(s)] for s in samps[scope]]
    end
    return reduce(hcat, [samps[k] for k in 1:length(samps)])
end

# Precompute which inds are missing in query so I can skip some steps and not always traverse.
function condsamp!(samps::Dict, node::ProductNode, inds, query)
    for c in children(node)
        condsamp!(samps, c, inds, query)
    end
end

function condsamp!(samps::Dict, node::SumNode, inds, query)
    pdfs = exp.([logpdf(c,query) for c in children(node)])
    w = pdfs .* weights(node)
    z = rand(Categorical(w / sum(w)), length(inds)) # Normalisation due to precision errors.
    # Generate observation by drawing from a child.
    for i in unique(z)
        condsamp!(samps, node[i], inds[i .== z], query)
    end
end

function condsamp!(samps::Dict, node::Leaf, inds, query)
    if ismissing(query[scope(node)])
        rand!(node.dist, view(samps[scope(node)], inds))
    end
end
