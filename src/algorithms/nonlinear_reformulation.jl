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
