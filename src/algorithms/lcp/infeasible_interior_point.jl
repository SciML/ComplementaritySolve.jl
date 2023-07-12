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

function __make_itp_linsolve_operator(M::AbstractMatrix,
    zₖ::AbstractVector,
    wₖ::AbstractVector)
    L = length(zₖ)
    @views function matvec(v::AbstractVector, u::AbstractVector, p, t)
        Δz, Δw = u[1:L], u[(L + 1):(2L)]
        v[1:L] .= M * Δz .- Δw
        v[(L + 1):(2L)] .= zₖ .* Δw .+ wₖ .* Δz
        return v
    end
    return FunctionOperator(matvec, similar(zₖ, 2L))
end

## For details see https://sites.math.washington.edu/~burke/crs/408f/notes/lcp/lcp.pdf
@views function solve(prob::LinearComplementarityProblem{false, false},
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
        A = __make_itp_linsolve_operator(M, z, w)
        b = vcat(-M * z .+ w .- q, σ * τ .- z .* w)
        Δzw = solve(LinearProblem(A, b), alg.linsolve; kwargs...)
        Δz, Δw = Δzw[1:length(z)], Δzw[(length(z) + 1):(2 * length(z))]

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
