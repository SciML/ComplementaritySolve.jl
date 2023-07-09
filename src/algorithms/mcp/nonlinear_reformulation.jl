
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
end
