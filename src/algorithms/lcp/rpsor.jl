# See: 
# https://nonsmooth.gricad-pages.univ-grenoble-alpes.fr/siconos/users_guide/problems_and_solvers/lcp.html#id6
function __error_estimate(::LinearComplementarityProblem, z, w)
    return norm(z - max.(0, z - w)) / sqrt(length(z))
end

# Regularized Projected Successive Overrelaxation (RPSOR) schemes for solving LCPs
Base.@kwdef struct RPSOR{sT} <: AbstractComplementarityAlgorithm
    ω::sT = 1.0
    ρ::sT = 0.0
    tol::sT = 1e-6
end

PSOR(ω=1.0; kwargs...) = RPSOR(; ω, kwargs...)
PGS(; kwargs...) = RPSOR(; ω=1.0, kwargs...)
RPGS(ρ=0.0; kwargs...) = RPSOR(; ω=1.0, ρ, kwargs...)

# RPSOR Algorithm as described in Section 12.4.6 (Acary, Brogliato 2008)
# zᵏ⁺¹ᵢ = max(0, zᵏᵢ − ω (Mᵢⱼ + ρ )⁻¹(qᵢ + ∑ⱼ Mᵢⱼ zᵏ⁺¹ⱼ + ∑ (Mᵢⱼ - ρ) zᵏⱼ))
function __solve(prob::LinearComplementarityProblem{iip, false}, alg::RPSOR, u0, M, q;
    maxiters=1000, kwargs...) where {iip}
    (; ω, ρ, tol) = alg

    z = copy(u0)
    zprev = similar(z)

    w_ρ = M * z .+ q
    err = tol

    for _ in 1:maxiters
        zprev .= z
        for i in eachindex(z)
            z[i] = max(0, z[i] - ω / (M[i, i] + ρ) * (w_ρ[i] - ρ * z[i]))
            w_ρ .+= view(M, :, i) * (z[i] - zprev[i])
            w_ρ[i] += ρ * (z[i] - zprev[i])
        end

        err = __error_estimate(prob, z, w_ρ)
        if err ≤ tol
            return LinearComplementaritySolution(z, [err], prob, alg, ReturnCode.Success)
        end
    end

    return LinearComplementaritySolution(z, [err], prob, alg, ReturnCode.MaxIters)
end
