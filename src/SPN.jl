module SPN

using HilbertSchmidtIndependenceCriterion,IntervalSets,StaticArrays,UUIDs,Distributions,NamedArrays

import Base: rand, getindex
import Distributions: scale, scale!, logpdf
#import StatsBase: countmap,corkendall,cor
import NamedArrays
import LightGraphs: connected_components, Graph
import Statistics: cor
import LinearAlgebra: diagind
import Combinatorics: permutations, combinations
import Clustering: kmeans
include("NegativeBinomial.jl")
include("nodes.jl")
include("show.jl")
include("probquerying.jl")
include("sampling.jl")
include("structurelearning.jl")
#include("optimization.jl")
include("helpers.jl")
end
