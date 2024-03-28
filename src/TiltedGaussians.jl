module TiltedGaussians
using LinearAlgebra,Statistics,Distributions,PDMats
include("quadrule.jl")
include("moments.jl")
export QuadRule,nnodes,tilted_moments
end # module TiltedGaussians
