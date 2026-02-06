using JuMP, SumOfSquares, DynamicPolynomials
using MosekTools
using Distributions, Combinatorics

function protect_replica(degree)
    barrier_degree, lagrange_degree = degree, degree

    dim = 2  # dimension of state space

    # Initial set
    L_initial = [2.75, 2.75]
    U_initial = [3.25, 3.25]

    # Unsafe set1
    L_unsafe1 = [0, 0]
    U_unsafe1 = [1, 1]

    # Unsafe set2
    L_unsafe2 = [9, 9]
    U_unsafe2 = [10, 10]

    # Combine unsafe regions
    unsafe_sets = [(L_unsafe1, U_unsafe1), (L_unsafe2, U_unsafe2)]

    # State space
    L_space = [0, 0]
    U_space = [10, 10]

    # ========================= Dynamics =========================
    NoiseType = "normal"
    sigma = [0.01, 0.01]
    mean = [0, 0]

    # Dynamics
    # f(x) = A * x + b + varsigma
    A = [0.90 0.10; 0.0 0.90]
    b = [0.45; -0.30]
    f(x, w) = A * x + b + w

    # Time horizon
    t = 10

    # Model
    model = Model(Mosek.Optimizer)
    @polyvar x[1:dim]  # state variables
    @polyvar w[1:dim]  # noise variables

    # Barrier polynomial
    monos = monomials(x, 0:barrier_degree)
    @variable(model, B, Poly(monos))
    @constraint(model, B in SOSCone())

    # B after one step of dynamics with analytical expectation over noise
    B_next = subs(B, x => f(x, w))
    # println(B_next)
    B_next = expectation_noise(B_next, sigma, w)
    # println(B_next)

    # Variables
    @variable(model, gamma)
    @constraint(model, gamma >= 0)
    @variable(model, c)
    @constraint(model, c >= 0)

    lambda = 10  # scaling factor
    @constraint(model, lambda - gamma - c * t >= 0)

    # Initial set constraints
    g0 = generate_polynomial(x, L_initial, U_initial)
    
    monos = monomials(x, 0:lagrange_degree)
    L0 = [
        @variable(model, variable_type=Poly(monos))
        for _ in 1:dim
    ]
    L0_times_g0 = sum(L0i * g0i for (L0i, g0i) in zip(L0, g0))
    
    for L0i in L0
        @constraint(model, L0i in SOSCone())
    end

    @constraint(model, -B + gamma - L0_times_g0 in SOSCone())
    
    # Unsafe set constraints
    g1s = []
    for (L_unsafe, U_unsafe) in unsafe_sets
        g1 = generate_polynomial(x, L_unsafe, U_unsafe)
        push!(g1s, g1)
    end

    monos = monomials(x, 0:lagrange_degree)
    L1s = []
    for _ in unsafe_sets
        L1 = []
        for _ in 1:dim
            L1i = @variable(model, variable_type=Poly(monos))
            push!(L1, L1i)
            @constraint(model, L1i in SOSCone())
        end

        push!(L1s, L1)
    end

    for (g1, L1) in zip(g1s, L1s)
        L1_times_g1 = sum(L * g for (L, g) in zip(L1, g1))
        @constraint(model, B - lambda - L1_times_g1 in SOSCone())
    end

    # State space constraints
    g = generate_polynomial(x, L_space, U_space)

    monos = monomials(x, 0:lagrange_degree)
    L = [
        @variable(model, variable_type=Poly(monos))
        for _ in 1:dim
    ]
    L_times_g = sum(Li * gi for (Li, gi) in zip(L, g))

    for Li in L
        @constraint(model, Li in SOSCone())
    end

    @constraint(model, -B_next + B + c - L_times_g in SOSCone())

    # Objective
    @objective(model, Min, (gamma + c * t) / lambda)

    optimize!(model)

    # println(summary_solution(model))
    gamma = value(gamma)
    c = value(c)
    return 1 - (gamma + c * t) / lambda, gamma, c
end

"""
Replace polynomial variable + SOS constraint with SOS variable.
"""
function modification1(degree)
    barrier_degree, lagrange_degree = degree, degree

    dim = 2  # dimension of state space

    # Initial set
    L_initial = [2.75, 2.75]
    U_initial = [3.25, 3.25]

    # Unsafe set1
    L_unsafe1 = [0, 0]
    U_unsafe1 = [1, 1]

    # Unsafe set2
    L_unsafe2 = [9, 9]
    U_unsafe2 = [10, 10]

    # Combine unsafe regions
    unsafe_sets = [(L_unsafe1, U_unsafe1), (L_unsafe2, U_unsafe2)]

    # State space
    L_space = [0, 0]
    U_space = [10, 10]

    # ========================= Dynamics =========================
    NoiseType = "normal"
    sigma = [0.01, 0.01]
    mean = [0, 0]

    # Dynamics
    # f(x) = A * x + b + varsigma
    A = [0.90 0.10; 0.0 0.90]
    b = [0.45; -0.30]
    f(x, w) = A * x + b + w

    # Time horizon
    t = 10

    # Model
    model = Model(Mosek.Optimizer)
    @polyvar x[1:dim]  # state variables
    @polyvar w[1:dim]  # noise variables

    # Barrier polynomial
    monos = monomials(x, 0:ceil(Int, barrier_degree/2))
    @variable(model, B, SOSPoly(monos))

    # B after one step of dynamics with analytical expectation over noise
    B_next = subs(B, x => f(x, w))
    B_next = expectation_noise(B_next, sigma, w)

    # Variables
    @variable(model, gamma)
    @constraint(model, gamma >= 0)
    @variable(model, c)
    @constraint(model, c >= 0)

    lambda = 10  # scaling factor
    @constraint(model, lambda - gamma - c * t >= 0)

    # Initial set constraints
    g0 = generate_polynomial(x, L_initial, U_initial)
    
    monos = monomials(x, 0:ceil(Int, lagrange_degree/2))
    L0 = [
        @variable(model, variable_type=SOSPoly(monos))
        for _ in 1:dim
    ]
    L0_times_g0 = sum(L0i * g0i for (L0i, g0i) in zip(L0, g0))
    @constraint(model, -B + gamma - L0_times_g0 in SOSCone())
    
    # Unsafe set constraints
    g1s = []
    for (L_unsafe, U_unsafe) in unsafe_sets
        g1 = generate_polynomial(x, L_unsafe, U_unsafe)
        push!(g1s, g1)
    end

    monos = monomials(x, 0:ceil(Int, lagrange_degree/2))
    L1s = []
    for _ in unsafe_sets
        L1 = []
        for _ in 1:dim
            L1i = @variable(model, variable_type=SOSPoly(monos))
            push!(L1, L1i)
        end

        push!(L1s, L1)
    end

    for (g1, L1) in zip(g1s, L1s)
        L1_times_g1 = sum(L * g for (L, g) in zip(L1, g1))
        @constraint(model, B - lambda - L1_times_g1 in SOSCone())
    end

    # State space constraints
    g = generate_polynomial(x, L_space, U_space)

    monos = monomials(x, 0:ceil(Int, lagrange_degree/2))
    L = [
        @variable(model, variable_type=SOSPoly(monos))
        for _ in 1:dim
    ]
    L_times_g = sum(Li * gi for (Li, gi) in zip(L, g))

    @constraint(model, -B_next + B + c - L_times_g in SOSCone())

    # Objective
    @objective(model, Min, (gamma + c * t) / lambda)

    optimize!(model)

    # println(summary_solution(model))
    gamma = value(gamma)
    c = value(c)
    return 1 - (gamma + c * t) / lambda, gamma, c
end

"""
Replace variable + non-negative constraint with constrained variable.
"""
function modification2(degree)
    barrier_degree, lagrange_degree = degree, degree

    dim = 2  # dimension of state space

    # Initial set
    L_initial = [2.75, 2.75]
    U_initial = [3.25, 3.25]

    # Unsafe set1
    L_unsafe1 = [0, 0]
    U_unsafe1 = [1, 1]

    # Unsafe set2
    L_unsafe2 = [9, 9]
    U_unsafe2 = [10, 10]

    # Combine unsafe regions
    unsafe_sets = [(L_unsafe1, U_unsafe1), (L_unsafe2, U_unsafe2)]

    # State space
    L_space = [0, 0]
    U_space = [10, 10]

    # ========================= Dynamics =========================
    NoiseType = "normal"
    sigma = [0.01, 0.01]
    mean = [0, 0]

    # Dynamics
    # f(x) = A * x + b + varsigma
    A = [0.90 0.10; 0.0 0.90]
    b = [0.45; -0.30]
    f(x, w) = A * x + b + w

    # Time horizon
    t = 10

    # Model
    model = Model(Mosek.Optimizer)
    @polyvar x[1:dim]  # state variables
    @polyvar w[1:dim]  # noise variables

    # Barrier polynomial
    monos = monomials(x, 0:ceil(Int, barrier_degree/2))
    @variable(model, B, SOSPoly(monos))

    # B after one step of dynamics with analytical expectation over noise
    B_next = subs(B, x => f(x, w))
    B_next = expectation_noise(B_next, sigma, w)

    # Variables
    @variable(model, gamma >= 0)
    @variable(model, c >= 0)

    lambda = 10  # scaling factor
    @constraint(model, lambda - gamma - c * t >= 0)

    # Initial set constraints
    g0 = generate_polynomial(x, L_initial, U_initial)
    
    monos = monomials(x, 0:ceil(Int, lagrange_degree/2))
    L0 = [
        @variable(model, variable_type=SOSPoly(monos))
        for _ in 1:dim
    ]
    L0_times_g0 = sum(L0i * g0i for (L0i, g0i) in zip(L0, g0))
    @constraint(model, -B + gamma - L0_times_g0 in SOSCone())
    
    # Unsafe set constraints
    g1s = []
    for (L_unsafe, U_unsafe) in unsafe_sets
        g1 = generate_polynomial(x, L_unsafe, U_unsafe)
        push!(g1s, g1)
    end

    monos = monomials(x, 0:ceil(Int, lagrange_degree/2))
    L1s = []
    for _ in unsafe_sets
        L1 = []
        for _ in 1:dim
            L1i = @variable(model, variable_type=SOSPoly(monos))
            push!(L1, L1i)
        end

        push!(L1s, L1)
    end

    for (g1, L1) in zip(g1s, L1s)
        L1_times_g1 = sum(L * g for (L, g) in zip(L1, g1))
        @constraint(model, B - lambda - L1_times_g1 in SOSCone())
    end

    # State space constraints
    g = generate_polynomial(x, L_space, U_space)

    monos = monomials(x, 0:ceil(Int, lagrange_degree/2))
    L = [
        @variable(model, variable_type=SOSPoly(monos))
        for _ in 1:dim
    ]
    L_times_g = sum(Li * gi for (Li, gi) in zip(L, g))

    @constraint(model, -B_next + B + c - L_times_g in SOSCone())

    # Objective
    @objective(model, Min, (gamma + c * t) / lambda)

    optimize!(model)

    # println(summary_solution(model))
    gamma = value(gamma)
    c = value(c)
    return 1 - (gamma + c * t) / lambda, gamma, c
end

"""
Only use relevant terms in Lagrange multipliers
"""
function modification3(degree)
    barrier_degree, lagrange_degree = degree, degree

    dim = 2  # dimension of state space

    # Initial set
    L_initial = [2.75, 2.75]
    U_initial = [3.25, 3.25]

    # Unsafe set1
    L_unsafe1 = [0, 0]
    U_unsafe1 = [1, 1]

    # Unsafe set2
    L_unsafe2 = [9, 9]
    U_unsafe2 = [10, 10]

    # Combine unsafe regions
    unsafe_sets = [(L_unsafe1, U_unsafe1), (L_unsafe2, U_unsafe2)]

    # State space
    L_space = [0, 0]
    U_space = [10, 10]

    # ========================= Dynamics =========================
    NoiseType = "normal"
    sigma = [0.01, 0.01]
    mean = [0, 0]

    # Dynamics
    # f(x) = A * x + b + varsigma
    A = [0.90 0.10; 0.0 0.90]
    b = [0.45; -0.30]
    f(x, w) = A * x + b + w

    # Time horizon
    t = 10

    # Model
    model = Model(Mosek.Optimizer)
    @polyvar x[1:dim]  # state variables
    @polyvar w[1:dim]  # noise variables

    # Barrier polynomial
    monos = monomials(x, 0:ceil(Int, barrier_degree/2))
    @variable(model, B, SOSPoly(monos))

    # B after one step of dynamics with analytical expectation over noise
    B_next = subs(B, x => f(x, w))
    B_next = expectation_noise(B_next, sigma, w)

    # Variables
    @variable(model, gamma >= 0)
    @variable(model, c >= 0)

    lambda = 10  # scaling factor
    @constraint(model, lambda - gamma - c * t >= 0)

    # Initial set constraints
    g0 = generate_polynomial(x, L_initial, U_initial)
    
    L0 = []
    for i in 1:dim
        monos = monomials(x[i], 0:ceil(Int, lagrange_degree/2))
        L0i = @variable(model, variable_type=SOSPoly(monos))
        push!(L0, L0i)
    end

    L0_times_g0 = sum(L0i * g0i for (L0i, g0i) in zip(L0, g0))
    @constraint(model, -B + gamma - L0_times_g0 in SOSCone())
    
    # Unsafe set constraints
    for (L_unsafe, U_unsafe) in unsafe_sets
        g1 = generate_polynomial(x, L_unsafe, U_unsafe)

        L1 = []
        for i in 1:dim
            monos = monomials(x[i], 0:ceil(Int, lagrange_degree/2))
            L1i = @variable(model, variable_type=SOSPoly(monos))
            push!(L1, L1i)
        end

        L1_times_g1 = sum(L * g for (L, g) in zip(L1, g1))
        @constraint(model, B - lambda - L1_times_g1 in SOSCone())
    end

    # State space constraints
    g = generate_polynomial(x, L_space, U_space)

    L = []
    for i in 1:dim
        monos = monomials(x[i], 0:ceil(Int, lagrange_degree/2))
        Li = @variable(model, variable_type=SOSPoly(monos))
        push!(L, Li)
    end

    L_times_g = sum(Li * gi for (Li, gi) in zip(L, g))
    @constraint(model, -B_next + B + c - L_times_g in SOSCone())

    # Objective
    @objective(model, Min, (gamma + c * t) / lambda)

    optimize!(model)

    # println(summary_solution(model))
    gamma = value(gamma)
    c = value(c)
    return 1 - (gamma + c * t) / lambda, gamma, c
end

"""
Use product of halfspaces + scalar Lagrange multipliers instead of SOS polynomials for Lagrange multipliers.
"""
function modification4(degree)
    barrier_degree, lagrange_degree = degree, degree

    dim = 2  # dimension of state space

    # Initial set
    L_initial = [2.75, 2.75]
    U_initial = [3.25, 3.25]

    # Unsafe set1
    L_unsafe1 = [0, 0]
    U_unsafe1 = [1, 1]

    # Unsafe set2
    L_unsafe2 = [9, 9]
    U_unsafe2 = [10, 10]

    # Combine unsafe regions
    unsafe_sets = [(L_unsafe1, U_unsafe1), (L_unsafe2, U_unsafe2)]

    # State space
    L_space = [0, 0]
    U_space = [10, 10]

    # ========================= Dynamics =========================
    NoiseType = "normal"
    sigma = [0.01, 0.01]
    mean = [0, 0]

    # Dynamics
    # f(x) = A * x + b + varsigma
    A = [0.90 0.10; 0.0 0.90]
    b = [0.45; -0.30]
    f(x, w) = A * x + b + w

    # Time horizon
    t = 10

    # Model
    model = Model(Mosek.Optimizer)
    @polyvar x[1:dim]  # state variables
    @polyvar w[1:dim]  # noise variables

    # Barrier polynomial
    monos = monomials(x, 0:ceil(Int, barrier_degree/2))
    @variable(model, B, SOSPoly(monos))

    # B after one step of dynamics with analytical expectation over noise
    B_next = subs(B, x => f(x, w))
    B_next = expectation_noise(B_next, sigma, w)

    # Variables
    @variable(model, gamma >= 0)
    @variable(model, c >= 0)

    lambda = 10  # scaling factor
    @constraint(model, lambda - gamma - c * t >= 0)

    # Initial set constraints
    g0 = generate_halfspaces(x, L_initial, U_initial)
    
    domain = sum(Iterators.product(g0, g0)) do (constraint1, constraint2)        
        # Lagragian multiplier
        τ = @variable(model, lower_bound=0.0)
        return τ * constraint1 * constraint2
    end

    @constraint(model, -B + gamma - domain in SOSCone())
    
    # Unsafe set constraints
    for (L_unsafe, U_unsafe) in unsafe_sets
        g1 = generate_halfspaces(x, L_unsafe, U_unsafe)

        domain = sum(Iterators.product(g1, g1)) do (constraint1, constraint2)        
            # Lagragian multiplier
            τ = @variable(model, lower_bound=0.0)
            return τ * constraint1 * constraint2
        end

        @constraint(model, B - lambda - domain in SOSCone())
    end

    # State space constraints
    g = generate_halfspaces(x, L_space, U_space)

    domain = sum(Iterators.product(g, g)) do (constraint1, constraint2)        
        # Lagragian multiplier
        τ = @variable(model, lower_bound=0.0)
        return τ * constraint1 * constraint2
    end

    @constraint(model, -B_next + B + c - domain in SOSCone())

    # Objective
    @objective(model, Min, (gamma + c * t) / lambda)

    optimize!(model)

    # println(summary_solution(model))
    gamma = value(gamma)
    c = value(c)
    return 1 - (gamma + c * t) / lambda, gamma, c
end

"""
Add unsafe outside state space constraint
"""
function modification5(degree)
    barrier_degree, lagrange_degree = degree, degree

    dim = 2  # dimension of state space

    # Initial set
    L_initial = [2.75, 2.75]
    U_initial = [3.25, 3.25]

    # Unsafe set1
    L_unsafe1 = [0, 0]
    U_unsafe1 = [1, 1]

    # Unsafe set2
    L_unsafe2 = [9, 9]
    U_unsafe2 = [10, 10]

    # Combine unsafe regions
    unsafe_sets = [(L_unsafe1, U_unsafe1), (L_unsafe2, U_unsafe2)]

    # State space
    L_space = [0, 0]
    U_space = [10, 10]

    # ========================= Dynamics =========================
    NoiseType = "normal"
    sigma = [0.01, 0.01]
    mean = [0, 0]

    # Dynamics
    # f(x) = A * x + b + varsigma
    A = [0.90 0.10; 0.0 0.90]
    b = [0.45; -0.30]
    f(x, w) = A * x + b + w

    # Time horizon
    t = 10

    # Model
    model = Model(Mosek.Optimizer)
    @polyvar x[1:dim]  # state variables
    @polyvar w[1:dim]  # noise variables

    # Barrier polynomial
    monos = monomials(x, 0:ceil(Int, barrier_degree/2))
    @variable(model, B, SOSPoly(monos))

    # B after one step of dynamics with analytical expectation over noise
    B_next = subs(B, x => f(x, w))
    B_next = expectation_noise(B_next, sigma, w)

    # Variables
    @variable(model, gamma >= 0)
    @variable(model, c >= 0)

    lambda = 10  # scaling factor
    @constraint(model, lambda - gamma - c * t >= 0)

    # Initial set constraints
    g0 = generate_halfspaces(x, L_initial, U_initial)
    
    domain = sum(Iterators.product(g0, g0)) do (constraint1, constraint2)        
        # Lagragian multiplier
        τ = @variable(model, lower_bound=0.0)
        return τ * constraint1 * constraint2
    end

    @constraint(model, -B + gamma - domain in SOSCone())
    
    # Unsafe set constraints
    for (L_unsafe, U_unsafe) in unsafe_sets
        g1 = generate_halfspaces(x, L_unsafe, U_unsafe)

        domain = sum(Iterators.product(g1, g1)) do (constraint1, constraint2)        
            # Lagragian multiplier
            τ = @variable(model, lower_bound=0.0)
            return τ * constraint1 * constraint2
        end

        @constraint(model, B - lambda - domain in SOSCone())
    end

    # State space constraints
    g = generate_halfspaces(x, L_space, U_space)

    domain = sum(Iterators.product(g, g)) do (constraint1, constraint2)        
        # Lagragian multiplier
        τ = @variable(model, lower_bound=0.0)
        return τ * constraint1 * constraint2
    end

    @constraint(model, -B_next + B + c - domain in SOSCone())

    # Unsafe outside state space constraint
    gu = generate_halfspaces(x, U_space, L_space)

    monos = monomials(x, 0:floor(Int, lagrange_degree / 2))
    for halfspace in gu
        lag_poly = @variable(model, variable_type=SOSPoly(monos))
        domain = lag_poly * halfspace
        @constraint(model, B - lambda - domain in SOSCone())
    end

    # Objective
    @objective(model, Min, (gamma + c * t) / lambda)

    optimize!(model)

    # println(summary_solution(model))
    gamma = value(gamma)
    c = value(c)
    return 1 - (gamma + c * t) / lambda, gamma, c
end

function generate_polynomial(x, L, U)
    return [(xi - Li) * (Ui - xi) for (xi, Li, Ui) in zip(x, L, U)]
end

"""
Generate halfspace constraints for a box defined by lower and upper bounds.
Halfspace constraints are of the form H @ x <= h, so xi - Li <= 0 and Ui - xi <= 0 for each dimension i.
"""
function generate_halfspaces(x, L, U)
    return vcat([xi - Li for (xi, Li) in zip(x, L)], [Ui - xi for (xi, Ui) in zip(x, U)])
end

function expectation_noise(expectation, standard_deviations, ws)
    exp = 0

    for term in terms(expectation)
        distributions = [Normal(0, sd) for sd in standard_deviations]
        w_degs = [MultivariatePolynomials.degree(term, w) for w in ws]
        w_expectation = sum(moment(dist, deg) for (dist, deg) in zip(distributions, w_degs))

        coeff_and_nonrand = subs(term, ws => ones(length(ws)))
        exp = exp + coeff_and_nonrand * w_expectation
    end

    return exp
end

function moment(dist::Normal, deg)
    @assert mean(dist) == 0 "Only zero-mean normal distributions are supported"

    if deg == 0
        return 1.0
    elseif isodd(deg)
        return 0.0
    else
        return Int64(doublefactorial(deg - 1)) * std(dist)^deg
    end
end