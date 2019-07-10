module SPN

using HilbertSchmidtIndependenceCriterion,IntervalSets,StaticArrays
using UUIDs,Distributions,IndexedTables,NamedArrays
using CategoricalArrays,RCall

import DataFrames
import Base: rand, getindex
import StatsBase: counts
import Distributions: scale, scale!, logpdf
import HypothesisTests: KruskalWallisTest, pvalue, PowerDivergenceTest
#import StatsBase: countmap,corkendall,cor
import NamedArrays
import LightGraphs: connected_components, Graph
import Statistics: cor
import LinearAlgebra: diagind
import Combinatorics: permutations, combinations
import Clustering: kmeans,kmedoids
include("NegativeBinomial.jl")
include("nodes.jl")
include("show.jl")
include("probquerying.jl")
include("sampling.jl")
include("structurelearning.jl")
#include("optimization.jl")
include("helpers.jl")
end
