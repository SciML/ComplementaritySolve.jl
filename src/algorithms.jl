abstract type AbstractComplementarityAlgorithm end

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

        _prob = NonlinearProblem(NonlinearFunction{true}(f!), prob.u0)
        sol = solve(_prob, alg.nlsolver, args...; kwargs...)

        return sol
    end
end
