struct AllMissing <: AbstractArray{Missing, 1}
end

logpdf(n::ProductNode, AM::AllMissing) = 0.
logpdf(n::SumNode, AM::AllMissing) = 0.


import Base.getindex, Base.size
getindex(AM::AllMissing, i::Integer) = missing
size(AM::AllMissing) = 0
setindex!(AM::AllMissing, v, i::Int) = @warn "Not an intended operation."; nothing
