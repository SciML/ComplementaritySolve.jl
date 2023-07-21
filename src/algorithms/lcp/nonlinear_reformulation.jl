for method in (:minmax, :smooth), B in (true, false)
    aType = NonlinearReformulation{method}
    op = Symbol("$(method)_transform")

    @eval function __solve(prob::LCP{iip, $B}, alg::$aType, u0, M, q; kwargs...) where {iip}
        f, θ = prob(M, q)

        residual_function = if iip
            function objective!(residual, u, θ)
                f(residual, u, θ)
                residual .= $(op).(residual, u)
                return residual
            end
        else
            objective(u, θ) = $(op).(f(u, θ), u)
        end

        _prob = NonlinearProblem(NonlinearFunction{iip}(residual_function), u0, θ)
        sol = solve(_prob, alg.nlsolver; kwargs...)
        return LinearComplementaritySolution(sol.u, sol.resid, prob, alg, sol.retcode)
    end
end
