abstract type AbstractComplementarityAlgorithm end

# Linear Complementarity Problem
## Works only if M is positive definite and symmetric
@kwdef struct BokhovenIterativeLCPAlgorithm{S} <: AbstractComplementarityAlgorithm
    nlsolver::S = NewtonRaphson()
end

@truncate_stacktrace BokhovenIterativeLCPAlgorithm

## NOTE: It is a steady state problem so we could in-principle use an ODE Solver
function solve(prob::LinearComplementarityProblem,
    alg::BokhovenIterativeLCPAlgorithm,
    args...;
    kwargs...)
    A = pinv(I + prob.M)
    B = A * (I - prob.M)
    b = -A * prob.q

    θ = vcat(vec(B), b)

    _get_B(θ) = reshape(view(θ, 1:length(B)), size(B))
    _get_b(θ) = view(θ, (length(B) + 1):length(θ))

    function objective!(residual, u, θ)
        mul!(residual, _get_B(θ), abs.(u))
        residual .+= _get_b(θ) .- u
        return residual
    end

    _prob = NonlinearProblem(NonlinearFunction{true}(objective!), prob.u0, θ)
    sol = solve(_prob, alg.nlsolver; kwargs...)

    z = abs.(sol.u)
    b .= z .- sol.u # |u| - u # This is `w`. Just reusing `b` to save memory
    z .+= sol.u     # |u| + u

    return LinearComplementaritySolution(z, b, sol.resid, prob, alg)
end

# Reformulate Problems as a Nonlinear Problem
struct NonlinearReformulation{method, S} <: AbstractComplementarityAlgorithm
    nlsolver::S
end

@truncate_stacktrace NonlinearReformulation 1

function NonlinearReformulation(method::Symbol=:smooth, nlsolver=NewtonRaphson())
    return NonlinearReformulation{method, typeof(nlsolver)}(nlsolver)
end

for method in (:minmax, :smooth)
    algType = NonlinearReformulation{method}
    op = Symbol("$(method)_transform")
    @eval function solve(prob::MixedComplementarityProblem{true}, alg::$algType; kwargs...)
        function f!(residual, u, θ)
            prob.f(residual, u, θ)
            residual .= $(op).(residual, u, prob.lb, prob.ub)
            return residual
        end

        _prob = NonlinearProblem(NonlinearFunction{true}(f!), prob.u0, prob.p)
        sol = solve(_prob, alg.nlsolver; kwargs...)

        return MixedComplementaritySolution(sol.u, sol.resid, prob, alg)
    end

    @eval function solve(prob::MixedComplementarityProblem{false}, alg::$algType; kwargs...)
        f(u, θ) = $(op).(prob.f(u, θ), u, prob.lb, prob.ub)

        _prob = NonlinearProblem(NonlinearFunction{false}(f), prob.u0, prob.p)
        sol = solve(_prob, alg.nlsolver; kwargs...)

        return MixedComplementaritySolution(sol.u, sol.resid, prob, alg)
    end

    @eval function solve(prob::LinearComplementarityProblem, alg::$algType; kwargs...)
        function f!(out, u, θ)
            M = reshape(view(θ, 1:length(prob.M)), size(prob.M))
            q = view(θ, (length(prob.M) + 1):length(θ))
            out .= q
            mul!(out, M, u, true, true)
            return out
        end

        function objective!(residual, u, θ)
            f!(residual, u, θ)
            residual .= $(op).(residual, u)
            return residual
        end

        θ = vcat(vec(prob.M), prob.q)
        _prob = NonlinearProblem(NonlinearFunction{true}(objective!), prob.u0, θ)
        sol = solve(_prob, alg.nlsolver; kwargs...)

        z = sol.u
        w = f!(copy(z), z, θ)

        return LinearComplementaritySolution(z, w, sol.resid, prob, alg)
    end
end

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
