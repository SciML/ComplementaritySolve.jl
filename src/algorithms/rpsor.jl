# See: 
# https://nonsmooth.gricad-pages.univ-grenoble-alpes.fr/siconos/users_guide/problems_and_solvers/lcp.html#id6
function error_estimate(::LinearComplementarityProblem, z, w)
    return norm(z - max.(0, z - w)) / sqrt(length(z))
end

# Regularized Projected Successive Overrelaxation (RPSOR) schemes for solving LCPs
Base.@kwdef struct RPSOR{sT} <: AbstractComplementarityAlgorithm
    ω::sT = 1.0
    ρ::sT = 0.0
    tol::sT = 1e-6
    maxiter::Int = 1000
end

PSOR(ω=1.0; kwargs...) = RPSOR(; ω, kwargs...)
PGS(; kwargs...) = RPSOR(; ω=1.0, kwargs...)
RPGS(ρ=0.0; kwargs...) = RPSOR(; ω=1.0, ρ, kwargs...)

# RPSOR Algorithm as described in Section 12.4.6 (Acary, Brogliato 2008)
# zᵏ⁺¹ᵢ = max(0, zᵏᵢ − ω (Mᵢⱼ + ρ )⁻¹(qᵢ + ∑ⱼ Mᵢⱼ zᵏ⁺¹ⱼ + ∑ (Mᵢⱼ - ρ) zᵏⱼ))
function solve(prob::LinearComplementarityProblem, alg::RPSOR; kwargs...)
    (; ω, ρ, tol, maxiter) = alg
    (; M, q, u0) = prob

    z = copy(u0)
    zprev = similar(z)

    w_ρ = M * z .+ q

    err = 2 * tol
    iter = 0

    while (err > tol && iter < maxiter)
        zprev .= z
        for i in eachindex(z)
            z[i] = max(0, z[i] - ω / (M[i, i] + ρ) * (w_ρ[i] - ρ * z[i]))
            w_ρ .+= view(M, :, i) * (z[i] - zprev[i])
            w_ρ[i] += ρ * (z[i] - zprev[i])
        end

        err = error_estimate(prob, z, w_ρ)
        iter += 1
    end

    w = w_ρ
    w .= M * z .+ q

    return LinearComplementaritySolution(z, w, err, prob, alg)
end
