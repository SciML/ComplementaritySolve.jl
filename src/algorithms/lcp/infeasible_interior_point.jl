# Infeasible Interior Point Method for monotone LCPs
@concrete struct InfeasibleInteriorPointMethod <: AbstractComplementarityAlgorithm
    linsolve
end

InfeasibleInteriorPointMethod() = InfeasibleInteriorPointMethod(nothing)

function __feasible_steplength(x, Δx; dims=:)
    T = eltype(x)
    z = x ./ Δx
    η = minimum(zᵢ -> ifelse(zᵢ ≥ 0, eltype(x)(Inf), -zᵢ), z; dims)
    return min(T(0.999) * η, T(1))
end

## For details see https://sites.math.washington.edu/~burke/crs/408f/notes/lcp/lcp.pdf
function solve(prob::LinearComplementarityProblem{false, false},
    alg::InfeasibleInteriorPointMethod;
    maxiters=1000,
    kwargs...)
    (; M, q, u0) = prob
    T = eltype(u0)

    ϵ = eps(T)
    σ = zero(T)

    z = copy(u0)
    w = min.(M * z .+ q, T(2))

    τ = dot(z, w)
    ρ = norm(M * z .+ q .- z, Inf)
    iter = 0

    while τ > ϵ || ρ > ϵ || iter ≤ maxiters
        # Linear Solve
        A = __make_block_matrix_operator([M -I(size(M, 1)); Diagonal(w) Diagonal(z)])

        η₁ = __feasible_steplength(z, Δz)
        η₂ = __feasible_steplength(w, Δw)

        z .+= η₁ .* Δz
        w .+= η₂ .* Δw

        τ = dot(z, w)
        ρ = norm(M * z .+ q .- z, Inf)

        σ = ifelse(τ ≤ ϵ && ρ > ϵ,
            T(1),
            min(T(0.5), (1 - η₁)^2, (1 - η₂)^2, abs(ρ - τ) / (ρ + 10 * τ)))

        iter += 1
    end
end
