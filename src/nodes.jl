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

import Base: getindex, keys, length

export ScopeMap
struct ScopeMap
    FeatureMap::T where T<:NamedTuple
    Types::T where T<:NamedTuple
end
function ScopeMap(nms::Tuple, types::Tuple)
    types = ifelse.((<:).(types, AbstractString), UInt32, types) # Categoricals repr as uints
    return ScopeMap(NamedTuple{nms}(collect(1:length(nms))), NamedTuple{Tuple(nms)}(types))
end

Base.keys(ScM::ScopeMap) = keys(ScM.FeatureMap)
Base.getindex(ScM::ScopeMap, i::Integer) = keys(ScM)[i]
Base.getindex(ScM::ScopeMap, nm::Symbol) = ScM.FeatureMap[nm]
Base.length(ScM::ScopeMap) = length(ScM.FeatureMap)


struct SumProductNetwork
    root::T where T<:Node
    categorical_pool::Dict{I,P} where {I<:Integer,P}
    ScM::ScopeMap
end

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
