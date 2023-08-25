struct RegionWithProbabilities{T, S<:LazySet{T}}
    region::S
    lower::Vector{T}
    upper::Vector{T}
    gap::Vector{T}

    sum_lower::T
    sum_upper::T

    function RegionWithProbabilities(region::S, transition_to_other_regions::Tuple{Vector{T}, Vector{T}}, transition_to_unsafe::Tuple{T, T}) where {T, S<:LazySet{T}}
        # Include custom constructor only for safety checks

        joint_lower_bound = sum(transition_to_other_regions[1]) + transition_to_unsafe[1]
        @assert joint_lower_bound <= 1 "The joint lower bound transition probability (is $joint_lower_bound) should be less than or equal to 1."

        joint_upper_bound = sum(transition_to_other_regions[2]) + transition_to_unsafe[2]
        @assert joint_upper_bound >= 1 - 1e-9 "The joint upper bound transition probability (is $joint_upper_bound) should be greater than or equal to 1."

        lower, upper = vcat(transition_to_other_regions[1], transition_to_unsafe[1]),
                       vcat(transition_to_other_regions[2], transition_to_unsafe[2])

        return new{T, S}(region, lower, upper, upper - lower, sum(lower), sum(upper))
    end
end

region(X::RegionWithProbabilities) = X.region
prob_lower(X::RegionWithProbabilities) = X.lower[1:end - 1]
prob_upper(X::RegionWithProbabilities) = X.upper[1:end - 1]
prob_unsafe_lower(X::RegionWithProbabilities) = X.lower[end]
prob_unsafe_upper(X::RegionWithProbabilities) = X.upper[end]

function ivi_prob(X::RegionWithProbabilities, p::AbstractVector{<:Int})
    v = Vector{Float64}(undef, length(p))
    return ivi_prob!(v, X, p)
end

function ivi_prob!(cache, X::RegionWithProbabilities, p::AbstractVector{<:Int})
    copyto!(cache, X.lower)
    
    remaining = 1 - X.sum_lower
    
    @inbounds for i in p
        cache[i] += min(remaining, X.gap[i])
        remaining -= X.gap[i]
        if remaining < 0
            break
        end
    end
    return cache
end

function update_regions(regions::Vector{<:RegionWithProbabilities}, p_distribution::Matrix{Float64})
    new_regions = Vector{RegionWithProbabilities}(undef, length(regions))

    Threads.@threads for jj in eachindex(regions)
        Xⱼ = regions[jj]
        p_values = p_distribution[:, jj]

        # Compute new transition probabilities
        new_other = prob_lower(Xⱼ), p_values[1:end - 1]

        # Compute new transition probabilities to unsafe region
        new_unsafe = prob_unsafe_lower(Xⱼ), p_values[end]

        new_regions[jj] = RegionWithProbabilities(region(Xⱼ), new_other, new_unsafe)
    end

    return new_regions
end

##### WARNING: This is type piracy #####
# This is a type piracy of the `LazySets` package.
# Please don't do this, but I needed to do this for the linear 2D system.
@commutative function LazySets.isdisjoint(H::AbstractHyperrectangle, B::Ball2)
    return LazySets._isdisjoint_convex_sufficient(H, B)
end