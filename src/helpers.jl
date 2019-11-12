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

# IndexedTables helpers

#Base.size(X::IndexedTable) = length(X), length(colnames(X))
#Base.size(X::IndexedTable, dim) = size(X)[dim]


# TypedTables Helpers
export dims, choose
choose(X::Table, nm::Symbol) = getproperty(X, nm)
choose(X::Table, nms::NTuple{T,Symbol}) where T = Table(NamedTuple{nms}([choose(X,nm) for nm in nms]))
choose(X::Table, nms::Vector{Symbol}) = choose(X::Table, Tuple(nms))
choose(X::Table, indices::NTuple{T,Integer}) where T = choose(X, columnnames(X)[collect(indices)])
choose(X::Table, indices::Vector{<:Integer}) = choose(X, columnnames(X)[indices])
choose(X::Table, index::Integer) = columns(X)[index]

dims(X::Table) = length(X), length(columnnames(X))
dims(X::Table, dim) = dims(X)[dim]
