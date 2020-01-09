# Non-mutating (recompile) update to SPN with parameters θ
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
