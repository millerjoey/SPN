export rand

##### Unconditional Sampling #####
rand(SPN::SumProductNetwork, n::Integer = 1) = rand(SPN, n, AllMissing())

#### Conditional Sampling #####

rand(SPN::SumProductNetwork, query::AbstractVector, n = 1) = rand(SPN::SumProductNetwork, n, query::AbstractVector)

# Allow query to be a dictionary. Dict("x1"=>5, "x2"=>1..2, "x5"=>"C") and assume the rest missing.
rand(SPN::SumProductNetwork, n::Integer, query::Dict) = rand(SPN, n, queryfromdict(SPN, query))

# Can possibly switch to use multithreading, making a single for loop.
function rand(SPN::SumProductNetwork, n::Integer, query)
    # force query to contain static arrays maybe?
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
    samps = Dict(i=>(ismissing(query[i]) ? convert(Vector{Union{T,Missing}}, fill(missing, n)) : fill(query[i], n)) for (i,T) in zip(1:length(SPN.ScM), SPN.ScM.Types))
    condsamp!(samps, SPN.root, 1:n, query)
    for (scope,pool) in SPN.categorical_pool
        samps[scope] = [pool.index[s] for s in samps[scope]]
    end
    return reduce(hcat, [samps[k] for k in 1:length(samps)])
end

# Precompute which inds are missing in query so I can skip some steps and not always traverse.
function condsamp!(samps, node::ProductNode, inds, query)
    for c in children(node)
        condsamp!(samps, c, inds, query)
    end
end

function condsamp!(samps, node::SumNode, inds, query)
    pdfs = exp.([logpdf(c,query) for c in children(node)])
    w = pdfs .* weights(node)
    z = rand(Categorical(w / sum(w)), length(inds)) # Normalisation due to precision errors.
    # Generate observation by drawing from a child.
    for i in unique(z)
        condsamp!(samps, node[i], inds[i .== z], query)
    end
end

function condsamp!(samps, node::Leaf, inds, query)
    if ismissing(query[scope(node)])
        samps[scope(node)][inds] .= rand(node.dist, length(inds))
    end
end
