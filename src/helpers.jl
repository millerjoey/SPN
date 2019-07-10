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
import Base: size
Base.size(X::IndexedTable) = length(X), length(colnames(X))
Base.size(X::IndexedTable, dim) = size(X)[dim]
