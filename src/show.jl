import Base: show

function Base.show(io::IO, N::Node)
    if N isa SumNode
        println(io, "SumNode\n\tweights = $(weights(N))")
    elseif N isa Leaf
        println(io, "Leaf\n\tdistribution = $(N.dist)")
    elseif N isa ProductNode
        println(io, "ProductNode")
    end
    println(io, "\tscope = $(scope(N))")
end
