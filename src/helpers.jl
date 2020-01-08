export scopeunion!, rpad!, scope, rescope!
export weights, logweights, scope, getindex, children
export size
function scopeunion!(A::BitArray, B::BitArray)
    if length(A) < length(B)
        rpad!(A, length(B))
    end
    A[1:length(B)] .= A[1:length(B)] .| B # Still problems here..
    return A
end

function scopeunion!(A::BitArray, n::Integer)
    rpad!(A, n)
    A[n] = true
    return A
end

function rpad!(A::BitArray, n::Integer)
    n > length(A) ? append!(A, falses(n - length(A))) : nothing
    return(A)
end

function rescope!(N::Node)
    for c in children(N)
        rescope!(c)
        scopeunion!(N.scope, c.scope)
    end
end
rescope!(L::Leaf) = nothing

weights(N::SumNode) = exp.(N.logweights)
logweights(N::SumNode) = N.logweights
children(N::Node) = N isa Leaf ? [] : N.children
scope(N::Leaf) = N.scope
scope(N::Node) = (1:length(N.scope))[N.scope]

getindex(N::Node, i::Integer) = N.children[i]

# TypedTables Helpers
export dims, select
select(X, nm::Symbol) = getproperty(X, nm)
select(X, nms::NTuple{T,Symbol}) where T = Table(NamedTuple{nms}([select(X,nm) for nm in nms]))
select(X, nms::Vector{Symbol}) = select(X::Table, Tuple(nms))
select(X, indices::NTuple{T,Integer}) where T = select(X, columnnames(X)[collect(indices)])
select(X, indices::Vector{<:Integer}) = select(X, columnnames(X)[indices])
select(X, index::Integer) = columns(X)[index]

dims(X::Table) = length(X), length(columnnames(X))
dims(X::Table, dim) = dims(X)[dim]

function queryfromdict(SPN::SumProductNetwork, query::Dict)
    if keytype(query) <: Symbol
        ks = keys(SPN.ScM)
    elseif keytype(query) <: Integer
        ks = 1:length(SPN.ScM)
    else
        @error("Keys of query must be columnnames as symbols or column position as integers.")
    end
    return Any[get(query,k,missing) for k in ks]
end
