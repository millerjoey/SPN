export compile, Node, SumNode, ProductNode, Leaf, SumProductNetwork, add!
export rescope!
export NB

abstract type Node end

struct SumNode <: Node
    logweights::Vector{Float64}
    children::Vector{T} where T<:Node
    scope::BitArray{1}
    id::S where S<:Integer
end
SumNode(logweights = Float64[], children = Node[], scope = BitVector([])) = SumNode(logweights, children, scope, uuid1().value)

struct ProductNode <: Node
    children::Vector{<:Node}
    scope::BitArray{1}
    id::S where S<:Integer
end

ProductNode(children::Vector{<:Node}, scope::BitVector) = ProductNode(children, scope, uuid1().value)
ProductNode() = ProductNode(Node[], BitVector([]))

struct Leaf{T<:Distribution} <: Node # How do I want to represent/compile indicators?
    dist::T
    scope::N where N <: Integer
    id::S where S<:Integer
end
Leaf(dist::Distribution, scope::Integer) = Leaf(dist, scope, uuid1().value)

struct SumProductNetwork
    root::T where T <:Node
end

# Builds cSPN with IDs. Returns initial parameters θ, contains function to map id -> θ subset.
function add!(N::SumNode, Child::Node, lw::Number)
    push!(N.children, Child)
    push!(N.logweights, lw)
    scopeunion!(N.scope, Child.scope)
    return N
end

function add!(N::ProductNode, Child::Node)
    push!(N.children, Child)
    scopeunion!(N.scope, Child.scope)
    return N
end

# Non-mutating (recompile) update to cSPN with parameters θ
function replace(SPN::SumProductNetwork, θ)
    parmap = SPN.parmap
    root = replace(SPN.root, θ, SPN.parmap)
    return CompiledSPN(root, parmap)
end

function replace(N::SumNode, θ, parmap)
    children = Tuple(replace(c, θ, parmap) for c in N.children)
    ϕ = θ[parmap[N.id]]
    return SumNode(ϕ/sum(ϕ), children, N.scope, N.id)
end

function replace(N::ProductNode, θ, parmap)
    children = Tuple(replace(c, θ, parmap) for c in N.children)
    return ProductNode(children, N.scope, N.id)
end

function replace(N::Leaf, θ, parmap)
    ϕ = θ[parmap[N.id]]
    Dtype = constructor(N.dist)
    D = safe_build_dist(Dtype, ϕ)
    return Leaf(D, N.scope, N.id)
end

function safe_build_dist(D::Type{Gamma}, ϕ)
    α = ϕ[1] + max(0.001 - ϕ[1], 0)
    θ = ϕ[2] + max(0.0001 - ϕ[2], 0)
    return D(α, θ)
end

function safe_build_dist(D::Type{Poisson}, ϕ)
    λ = ϕ[1] + max(0.0001 - ϕ[1], 0)
    return D(λ)
end

function safe_build_dist(D::Type{NB}, ϕ)
    r = ϕ[1] + max(0.0001 - ϕ[1], 0)
    p = ϕ[2]
    return D(r, p)
end


constructor(D::Normal) = Normal
constructor(D::Gamma) = Gamma
constructor(D::Exponential) = Exponential
constructor(D::NegativeBinomial) = NB
constructor(D::Poisson) = Poisson
