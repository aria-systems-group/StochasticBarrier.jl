""" Functions to compute :

    Transition probability bounds P̲ᵢⱼ ≤ P(f(x) ∈ qᵢ | x ∈ qⱼ) ≤ P̅ᵢⱼ for Linear Systems and Neural Network Dynamic Models

    © Rayan Mazouz, Frederik Baymler Mathiesen

"""
abstract type AbstractLowerBoundAlgorithm end
struct VertexEnumeration <: AbstractLowerBoundAlgorithm end

abstract type AbstractUpperBoundAlgorithm end
Base.@kwdef struct GlobalSolver <: AbstractUpperBoundAlgorithm
    non_linear_solver = default_non_linear_solver()
end
struct BoxApproximation <: AbstractUpperBoundAlgorithm end
Base.@kwdef struct FrankWolfe <: AbstractUpperBoundAlgorithm
    linear_solver = default_lp_solver()
    sdp_solver = default_sdp_solver()
    num_iterations = 100
    termination_ϵ = 1e-12
end

Base.@kwdef struct TransitionProbabilityAlgorithm
    lower_bound_method::AbstractLowerBoundAlgorithm = VertexEnumeration()
    upper_bound_method::AbstractUpperBoundAlgorithm = GlobalSolver()
    sparisty_ϵ = 1e-12
end

transition_probabilities(system::AdditiveGaussianUncertainPWASystem; kwargs...) = transition_probabilities(system, regions(system); kwargs...)
function transition_probabilities(system, Xs; alg=TransitionProbabilityAlgorithm())
    # Construct barriers
    @info "Computing transition probabilities"

    safe_set = Hyperrectangle(low=minimum(low.(Xs)), high=maximum(high.(Xs)))
    # TODO: Check if ⋃Xs = state_space

    # Anything beyond `μ ± σ * nσ_search` will always have a probability < sparisty_ϵ
    nσ_search = -quantile(Normal(), alg.sparisty_ϵ)

    # Size definition
    number_hypercubes = length(Xs)

    # Pre-allocate probability matrices
    P̲ = Vector{SparseVector{Float64, Int64}}(undef, number_hypercubes)
    P̅ = Vector{SparseVector{Float64, Int64}}(undef, number_hypercubes)

    # Generate
    bar = Progress(number_hypercubes)
    Threads.@threads for jj in eachindex(Xs)
        P̲ⱼ, P̅ⱼ = transition_prob_from_region(system, (jj, Xs[jj]), Xs, safe_set, alg; nσ_search=nσ_search)

        P̲[jj] = P̲ⱼ
        P̅[jj] = P̅ⱼ
        
        next!(bar)
    end

    # Combine into a single matrix
    P̲ = reduce(sparse_hcat, P̲)
    P̅ = reduce(sparse_hcat, P̅)

    density = (nnz(P̅) + nnz(P̲)) / (length(P̅) + length(P̲))
    @info "Density of the probability matrix" density sparsity=1-density

    axlist = (Dim{:to}(1:number_hypercubes + 1), Dim{:from}(1:number_hypercubes))
    P̲, P̅ = DimArray(P̲, axlist), DimArray(P̅, axlist)

    # Return as a YAXArrays dataset
    return create_sparse_probability_dataset(Xs, P̲, P̅)
end

function post(system::AdditiveGaussianLinearSystem, Xind)
    (jj, X) = Xind

    X = convert(VPolytope, X)

    # Compute post(qᵢ, f(x)) for all qⱼ ∈ Q
    A, b = dynamics(system)

    VY = affine_map(A, X, b)
    HY = convert(HPolytope, VY)
    box_Y = box_approximation(VY)

    return VY, HY, box_Y
end

function post(system::AdditiveGaussianUncertainPWASystem, Xind)
    (jj, X) = Xind

    # Compute post(qᵢ, f(x)) for all qⱼ ∈ Q    
    (Xprime, dyn) = dynamics(system)[jj]
    # This is a necessary but not suffiecient condition.
    # The complete condition is that the intersection of the two regions has a non-zero measure.
    @assert !isdisjoint(X, Xprime)

    X = convert(VPolytope, X)
    VY = VPolytope(mapreduce(vcat, dyn) do (A, b)
        vertices_list(affine_map(A, X, b))
    end)

    HY = convert(HPolytope, VY)
    box_Y = box_approximation(VY)

    return VY, HY, box_Y
end

# Transition probability P̲ᵢⱼ ≤ P(f(x) ∈ qᵢ | x ∈ qⱼ) ≤ P̅ᵢⱼ based on proposition 1, http://dx.doi.org/10.1145/3302504.3311805
function transition_prob_from_region(system, Xⱼ, Xs, safe_set, alg; nσ_search)
    VY, HY, box_Y = post(system, Xⱼ)

    # Fetch noise
    n = length(Xs)
    σ = noise_distribution(system)

    # Search for overlap with box(f(Xⱼ)) + σ * nσ_search as 
    # any region beyond that will always have a probability < sparisty_ϵ
    query_set = minkowski_sum(box_Y, Hyperrectangle(zero(σ), σ * nσ_search)) 

    indices = findall(X -> !isdisjoint(X, query_set), Xs)

    P̲ⱼ = zeros(Float64, length(indices))
    P̅ⱼ = zeros(Float64, length(indices))

    for (i, Xᵢ) in enumerate(@view(Xs[indices]))
        # Obtain min and max of T(qᵢ | x) over Y
        P̲ᵢⱼ, P̅ᵢⱼ = transition_prob_to_region(system, VY, HY, box_Y, Xᵢ, alg)
        
        P̲ⱼ[i] = P̲ᵢⱼ
        P̅ⱼ[i] = P̅ᵢⱼ
    end

    # Prune regions with P(f(x) ∈ qᵢ | x ∈ qⱼ) < sparisty_ϵ
    keep_indices = findall(p -> p >= alg.sparisty_ϵ, P̅ⱼ)
    P̲ⱼ = P̲ⱼ[keep_indices]
    P̅ⱼ = P̅ⱼ[keep_indices]
    indices = indices[keep_indices]

    # Compute P(f(x) ∈ qᵤ | x ∈ qⱼ) including sparsity pruning
    P̲ₛⱼ, P̅ₛⱼ = transition_prob_to_region(system, VY, HY, box_Y, safe_set, alg)
    Psparse = (n - length(indices)) * alg.sparisty_ϵ
    P̲ᵤⱼ, P̅ᵤⱼ = (1 - P̅ₛⱼ), (1 - P̲ₛⱼ) + Psparse
    
    # If you ever hit this case, then you are in trouble. Either sparisty_ϵ
    # is too high for the amount of regions, or the system is inherently unsafe.
    # Relaxing the assert with δ = 1e-6
    @assert P̅ᵤⱼ <= 1.0 + 1e-6

    # Clipping P̅ᵤⱼ @ 1
    P̅ᵤⱼ = min(P̅ᵤⱼ, 1.0)

    P̲ⱼ = SparseVector(n + 1, [indices; [n + 1]], [P̲ⱼ; [P̲ᵤⱼ]])
    P̅ⱼ = SparseVector(n + 1, [indices; [n + 1]], [P̅ⱼ; [P̅ᵤⱼ]])

    # Enforce consistency (this is useful particularly with BoxApproximation)
    P̲ⱼ, P̅ⱼ = enforce_consistency(P̲ⱼ, P̅ⱼ)

    return P̲ⱼ, P̅ⱼ
end

# Transition probability P̲ᵢⱼ ≤ P(f(x) ∈ qᵢ | x ∈ qⱼ) ≤ P̅ᵢⱼ based on proposition 1, http://dx.doi.org/10.1145/3302504.3311805
function transition_prob_to_region(system, VY, HY, box_Y, Xᵢ, alg)
    v = LazySets.center(Xᵢ)
    σ = noise_distribution(system)
    
    # Transition kernel T(qᵢ | x)
    kernel = GaussianTransitionKernel(Xᵢ, σ)
    log_kernel = GaussianLogTransitionKernel(Xᵢ, σ)

    # Obtain min of T(qᵢ | x) over Y
    prob_transition_lower = min_log_concave_over_polytope(alg.lower_bound_method, kernel, v, VY)

    # Obtain max of T(qᵢ | x) over Y
    prob_transition_upper = exp(max_quasi_concave_over_polytope(alg.upper_bound_method, log_kernel, v, HY, box_Y))

    return prob_transition_lower, prob_transition_upper
end

struct GaussianLogTransitionKernel{S, VS<:AbstractVector{S}, H<:AbstractHyperrectangle{S}}
    X::H
    σ::VS
end

function (T::GaussianLogTransitionKernel)(y)
    m = LazySets.dim(T.X)
    vₗ, vₕ = low(T.X), high(T.X)

    acc = log(1) - m * log(2) + mapreduce(+, y, vₗ, vₕ, T.σ) do yᵢ, vₗᵢ, vₕᵢ, σᵢ
        a = invsqrt2 * (yᵢ - vₗᵢ) / σᵢ
        b = invsqrt2 * (yᵢ - vₕᵢ) / σᵢ
        logerf(b, a)   # log(erf(a) - erf(b))
    end

    return acc
end

struct GaussianTransitionKernel{S, VS<:AbstractVector{S}, H<:AbstractHyperrectangle{S}}
    log_kernel::GaussianLogTransitionKernel{S, VS, H}
end
function GaussianTransitionKernel(X, σ)
    return GaussianTransitionKernel(GaussianLogTransitionKernel(X, σ))
end

function (T::GaussianTransitionKernel)(y)
    # This is more numerically stable than computing it directly
    return exp(T.log_kernel(y))
end

function min_log_concave_over_polytope(::VertexEnumeration, f, global_max, X)
    vertices = vertices_list(X)

    return minimum(f, vertices)

    # convex_center = sum(vertices)
    # convex_center ./= length(vertices)

    # dir = global_max - convex_center

    # lb = 1.0
    # for v in vertices
    #     if dot(v - convex_center, dir) <= 0
    #         lb = min(lb, f(v))
    #     end
    # end

    # return lb
end

function max_quasi_concave_over_polytope(::BoxApproximation, f, global_max, X, box_X)
    if global_max in X
        return f(global_max)
    end

    x_max = project_onto_hyperrect(box_X, global_max)
    return f(x_max)
end

function project_onto_hyperrect(X, p)
    l, h = low(X), high(X)
    return @. min(h, max(p, l))
end

function max_quasi_concave_over_polytope(alg::GlobalSolver, f, global_max, X, box_X)
    if global_max in X
        return f(global_max)
    end

    m = LazySets.dim(X)
    fsplat(y...) = f(y)

    model = Model(alg.non_linear_solver)
    set_silent(model)
    register(model, :fsplat, m, fsplat; autodiff = true)

    @variable(model, x[1:m])

    H, h = tosimplehrep(X)
    @constraint(model, H * x <= h)

    @NLobjective(model, Max, fsplat(x...))

    # Optimize for maximum
    JuMP.optimize!(model)

    return JuMP.objective_value(model) + 1e-8  # Account for covergence tolerance
end

function max_quasi_concave_over_polytope(alg::FrankWolfe, f, global_max, X, box_X)
    if global_max in X
        return f(global_max)
    end

    # Frequently, the closest point to the global maximum is maximum within the polytope.
    x_cur = l2_closest_point(X, global_max, alg.sdp_solver)

    x_min = similar(x_cur)
    d = similar(x_cur)
    ∇ₓf = similar(x_cur)

    model = get!(() -> Model(alg.linear_solver), task_local_storage(), "transition_prob_lp_model")
    set_silent(model)
    empty!(model)

    @variable(model, x[eachindex(x_cur)])

    H, h = tosimplehrep(X)
    @constraint(model, H * x <= h)

    for k in 0:alg.num_iterations
        # Compute gradient
        ∇ₓf .= -gradient(f, x_cur)[1]   # We can accelerate this more if we manually derive the gradient
        x_min .= compute_extreme_point(model, ∇ₓf, x)

        d .= x_min .- x_cur
        dual_grap = -dot(∇ₓf, d)

        if dual_grap < alg.termination_ϵ
            break
        end

        x_cur .+= (8 / (k + 8)) .* d  # Agnostic step size with l = 8
    end
    
    return f(x_cur) / (1 + 4 * 10^(log10(alg.termination_ϵ) / 3)) # Account for covergence tolerance
end

function compute_extreme_point(model, ∇ₓf, x)
    @objective(model, Min, dot(∇ₓf, x))
    JuMP.optimize!(model)

    return JuMP.value.(x)
end

function l2_closest_point(X, p, sdp_solver)
    H, h = tosimplehrep(X)

    model = get!(() -> Model(sdp_solver), task_local_storage(), "l2_closest_point_model")
    set_silent(model)
    empty!(model)

    @variable(model, x[1:LazySets.dim(X)])
    @constraint(model, H * x <= h)
    @objective(model, Min, sum((x - p).^2))
    optimize!(model)

    return JuMP.value.(x)
end

function enforce_consistency(P̲, P̅)
    # Enforce consistency
    sum_lower = 1 - sum(P̲)
    P̅ = min.(P̅, sum_lower .+ P̲)

    return P̲, P̅
end

plot_posterior(system::AdditiveGaussianUncertainPWASystem; kwargs...) = plot_posterior(system, regions(system); kwargs...)
function plot_posterior(system, Xs; figname_prefix="")
    pwa_dynamics = dynamics(system)

    VYs = map(pwa_dynamics) do (X, dyn)
        X = convert(VPolytope, X)

        Y = map(dyn) do (A, b)
            affine_map(A, X, b)
        end
        return Y
    end

    postXs = post.(tuple(system), zip(eachindex(Xs), Xs))

    box_Ys = map(postXs) do (VY, HY, box_Y)
        box_Y
    end

    for (i, (X, Y, box_Y)) in enumerate(zip(Xs[1:10], VYs, box_Ys))
        p = plot(X, color=:blue, alpha=0.2, xlim=(-deg2rad(15), deg2rad(15)), ylim=(-1.2, 1.2), size=(1200, 800))
        plot!(p, Y[1], color=:red, alpha=0.2)
        plot!(p, Y[2], color=:red, alpha=0.2)
        plot!(p, box_Y, color=:green, alpha=0.2)

        savefig(p, figname_prefix * "posterior_$i.png")
    end
end