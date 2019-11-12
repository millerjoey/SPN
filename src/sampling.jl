export rand

##### Unconditional Sampling #####
rand(SPN::SumProductNetwork) = rand(SPN::SumProductNetwork, 1)
# Can probably optimize for large n; don't need to repeat traversal

function rand(SPN::SumProductNetwork, n::Integer)
    samps = Matrix(undef, n, length(SPN.root.scope))
    for i in 1:n
        uncondsamp!(SPN.root, view(samps, i, :))
    end
    for (scope,pool) in SPN.categorical_pool
        samps[:, scope] .= [pool.index[s] for s in samps[:, scope]]
    end
    return samps
end

function uncondsamp!(node::ProductNode, samp)
    for c in children(node)
        uncondsamp!(c, samp)
    end
end

function uncondsamp!(node::SumNode, samp)
    w = weights(node)
    z = rand(Categorical(w / sum(w)))
    uncondsamp!(node[z], samp)
end

function uncondsamp!(node::Leaf, samp)
    samp[scope(node)] = rand(node.dist)
end

#### Conditional Sampling #####

rand(SPN::SumProductNetwork, query::AbstractVector, n = 1) = rand(SPN::SumProductNetwork, n, query::AbstractVector)
# Can optimize for large n; don't need to repeat traversal

function rand(SPN::SumProductNetwork, n::Integer, query::AbstractVector)
    q_orig = deepcopy(query)
    for (scope,pool) in SPN.categorical_pool
        query[scope] isa Missing ? continue : nothing
        if isa(query[scope], AbstractVector)
            query[scope] = [pool.invindex[v] for v in query[scope]]
        elseif isa(query[scope], String)
            query[scope] = pool.invindex[query[scope]]
        end
    end
    query = repeat(permutedims(convert(Vector{Any}, query)), outer = [n,1])
    for i in 1:n
        condsamp!(SPN.root, view(query, i, :))
    end
    for scope in keys(SPN.categorical_pool)
        if !(q_orig[scope] isa Missing)
            for i in 1:size(query, 1)
                query[i, scope] = q_orig[scope]
            end
        end
    end
    return query
end

function condsamp!(node::ProductNode, query)
    for c in children(node)
        condsamp!(c, query)
    end
end

function condsamp!(node::SumNode, query)
    pdfs = exp.([logpdf(c,query) for c in children(node)])
    w = pdfs .* weights(node)
    z = rand(Categorical(w / sum(w))) # Normalisation due to precision errors.
    # Generate observation by drawing from a child.
    condsamp!(node[z], query)
end

function condsamp!(node::Leaf, query)
    if !isa(query[scope(node)],Interval) && ismissing(query[scope(node)])
        query[scope(node)] = rand(node.dist)
    end
end
