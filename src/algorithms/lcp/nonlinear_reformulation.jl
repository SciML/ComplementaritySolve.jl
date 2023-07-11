for method in (:minmax, :smooth), batched in (false, true)
    algType = NonlinearReformulation{method}
    op = Symbol("$(method)_transform")

    @eval function solve(prob::LinearComplementarityProblem{iip, $batched},
        alg::$algType;
        kwargs...) where {iip}
        f, u0, θ = prob()

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

        z = sol.u
        w = iip ? f(similar(z), z, θ) : f(z, θ)

        if $(batched)
            z = dropdims(z; dims=2)
            w = dropdims(w; dims=2)
        end

        return LinearComplementaritySolution(z, w, sol.resid, prob, alg)
    end
end
