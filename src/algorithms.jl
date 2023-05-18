abstract type AbstractComplementarityAlgorithm end

# Linear Complementarity Problem
## Works only if M is positive definite and symmetric
@kwdef struct BokhovenIterativeLCPAlgorithm{S} <: AbstractComplementarityAlgorithm
    nlsolver::S = NewtonRaphson()
end

## NOTE: It is a steady state problem so we could in-principle use an ODE Solver
function solve(prob::LinearComplementarityProblem, alg::BokhovenIterativeLCPAlgorithm,
               args...; kwargs...)
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
    sol = solve(_prob, alg.nlsolver, args...; kwargs...)

    z = abs.(sol.u)
    b .= z .- sol.u # |u| - u # This is `w`. Just reusing `b` to save memory
    z .+= sol.u     # |u| + u

    return LinearComplementaritySolution(z, b, sol.resid, prob, alg)
end

# Reformulate Problems as a Nonlinear Problem
struct NonlinearReformulation{method, S} <: AbstractComplementarityAlgorithm
    nlsolver::S
end

function NonlinearReformulation(method::Symbol=:smooth, nlsolver=NewtonRaphson())
    return NonlinearReformulation{method, typeof(nlsolver)}(nlsolver)
end

for method in (:minmax, :smooth)
    algType = NonlinearReformulation{method}
    op = Symbol("$(method)_transform")
    @eval function solve(prob::MixedComplementarityProblem{true}, alg::$algType, args...;
                         kwargs...)
        function f!(residual, u, θ)
            prob.f(residual, u, θ)
            residual .= $(op).(residual, u, prob.lb, prob.ub)
            return residual
        end

        _prob = NonlinearProblem(NonlinearFunction{true}(f!), prob.u0, prob.p)
        sol = solve(_prob, alg.nlsolver, args...; kwargs...)

        return sol
    end

    @eval function solve(prob::MixedComplementarityProblem{false}, alg::$algType, args...;
                         kwargs...)
        f(u, θ) = $(op).(prob.f(u, θ), u, prob.lb, prob.ub)

        _prob = NonlinearProblem(NonlinearFunction{false}(f), prob.u0, prob.p)
        sol = solve(_prob, alg.nlsolver, args...; kwargs...)

        return sol
    end

    @eval function solve(prob::LinearComplementarityProblem, alg::$algType, args...;
                         kwargs...)
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
        sol = solve(_prob, alg.nlsolver, args...; kwargs...)

        z = sol.u
        w = f!(copy(z), z, θ)

        return LinearComplementaritySolution(z, w, sol.resid, prob, alg)
    end
end
