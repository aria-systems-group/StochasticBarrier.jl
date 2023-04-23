module PiecewiseBarrier

using SumOfSquares
using MultivariatePolynomials, DynamicPolynomials
const MP = MultivariatePolynomials

using MosekTools
using LazySets
using StatsBase
using Combinatorics
using LinearAlgebra
using GLPK
using Optim
using JuMP
import JuMP.@variable
using LaTeXStrings

include("constants.jl")

include("utility.jl")
export state_space_generation, vectorize

include("system.jl")
export AbstractDiscreteTimeStochasticSystem, AdditiveGaussianPolynomialSystem
export variables, dynamics, noise_distribution

include("expectation.jl")

include("exponential.jl")
export exponential_bounds

include("validate.jl")

include("sumofsquares.jl")
export sos_barrier

include("barrier.jl")

include("piecewise.jl")
export piecewise_barrier

end # module PiecewiseBarrier
