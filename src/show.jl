import Base: show

function Base.show(io::IO, N::Node)
    if N isa SumNode
        println(io, "\tweights = $(weights(N))")
    end
    if N isa Leaf
        println(io, "\tdistribution = $(N.dist)")
    end
    println(io, "\tscope = $(scope(N))")
end
