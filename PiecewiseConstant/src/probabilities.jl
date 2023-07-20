""" Functions to compute :

    Transition probability bounds P̲ᵢⱼ ≤ P(f(x) ∈ qᵢ | x ∈ qⱼ) ≤ P̅ᵢⱼ for Linear Systems and Neural Network Dynamic Models

    © Rayan Mazouz, Frederik Baymler Mathiesen

"""

transition_probabilities(system::AdditiveGaussianUncertainPWASystem) = transition_probabilities(system, regions(system))
function transition_probabilities(system, Xs)

    # Construct barriers
    @info "Computing transition probabilities"

    # Size definition
    number_hypercubes = length(Xs)

    # Compute post(qⱼ, f(x)) for all qⱼ ∈ Q
    Ys, box_Ys = post(system, Xs)

    # Pre-allocate probability matrices
    P̲ = zeros(number_hypercubes, number_hypercubes)
    P̅ = zeros(number_hypercubes, number_hypercubes)

    # Generate
    Threads.@threads for ii in eachindex(Xs)
        P̲ᵢ, P̅ᵢ = transition_prob_to_region(system, Ys, box_Ys, Xs[ii])

        P̲[ii, :] = P̲ᵢ
        P̅[ii, :] = P̅ᵢ
    end

    Xₛ = Hyperrectangle(low=minimum(low.(Xs)), high=maximum(high.(Xs)))

    P̲ₛ, P̅ₛ  = transition_prob_to_region(system, Ys, box_Ys, Xₛ)
    P̲ᵤ, P̅ᵤ = (1 .- P̅ₛ), (1 .- P̲ₛ)

    # Return as a tuple
    return P̲, P̅, P̲ᵤ, P̅ᵤ
end

function post(system::AdditiveGaussianLinearSystem, Xs)
    # Compute post(qᵢ, f(x)) for all qⱼ ∈ Q
    A, b = dynamics(system)
    f(x) = A * x + b

    Xs = convert.(VPolytope, Xs)
    Ys = f.(Xs)
    box_Ys = box_approximation.(Ys)

    return Ys, box_Ys
end

function post(system::AdditiveGaussianUncertainPWASystem, Xs)
    # Input Xs is also contained in dynamics(system) since _piece-wise_ affine.

    # Compute post(qᵢ, f(x)) for all qⱼ ∈ Q    
    pwa_dynamics = dynamics(system)

    Ys = map(pwa_dynamics) do (X, dyn)
        X = convert(VPolytope, X)

        vertices = mapreduce(vcat, dyn) do (A, b)
            vertices_list(A * X + b)
        end
        return VPolytope(vertices)
    end
    box_Ys = box_approximation.(Ys)

    return Ys, box_Ys
end

# Transition probability P̲ᵢⱼ ≤ P(f(x) ∈ qᵢ | x ∈ qⱼ) ≤ P̅ᵢⱼ based on proposition 1, http://dx.doi.org/10.1145/3302504.3311805
function transition_prob_to_region(system, Ys, box_Ys, Xᵢ)
    vₗ = low(Xᵢ)
    vₕ = high(Xᵢ)
    v = center(Xᵢ)

    # Fetch noise
    m = dimensionality(system)
    σ = noise_distribution(system)
    
    # Transition kernel T(qᵢ | x)
    erf_lower(y, i) = erf((y[i] - vₗ[i]) / (σ[i] * sqrt(2)))
    erf_upper(y, i) = erf((y[i] - vₕ[i]) / (σ[i] * sqrt(2)))
    T(y) = (1 / 2^m) * prod(i -> erf_lower(y, i) - erf_upper(y, i), 1:m)

    # Obtain min of T(qᵢ | x) over Ys
    prob_transition_lower = map(Ys) do Y
        vertices = vertices_list(Y)

        P_min = minimum(T, vertices)
        return P_min
    end

    # Obtain max of T(qᵢ | x) over Ys
    prob_transition_upper = map(box_Ys) do Y
        if v in Y
            return T(v)
        end

        l, h = low(Y), high(Y)

        y_max = @. min(h, max(v, l))

        P_max = T(y_max)
        return P_max
    end

    return prob_transition_lower, prob_transition_upper
end
