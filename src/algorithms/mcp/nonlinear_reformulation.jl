for method in (:minmax, :smooth)
    algType = NonlinearReformulation{method}
    op = Symbol("$(method)_transform")
    @eval function __solve(prob::MCP{true}, alg::$algType, u0, p; kwargs...)
        function f!(residual, u, θ)
            prob.f(residual, u, θ)
            residual .= $(op).(residual, u, prob.lb, prob.ub)
            return residual
        end

        function jac!(J,u,θ)
            function _f!(F,u)
                f!(F,u,θ)
                return F
            end
            residual = similar(u)
            jac = Float64.(Symbolics.jacobian_sparsity(f!,residual,u,θ))
            colors = SparseDiffTools.matrix_colors(jac)
            FiniteDiff.finite_difference_jacobian!(J,_f!,u,colorvec=colors,sparsity=jac)
        end

        _prob = NonlinearProblem(NonlinearFunction{true}(f!;jac=jac!), u0, p)
        sol = solve(_prob, alg.nlsolver; kwargs...)

        return MixedComplementaritySolution(sol.u, sol.resid, prob, alg, sol.retcode)
    end

    @eval function __solve(prob::MCP{false}, alg::$algType, u0, p; kwargs...)
        f(u, θ) = $(op).(prob.f(u, θ), u, prob.lb, prob.ub)
        
        function jac(u,θ)
            function _f!(F,u,θ)
                F .= f(u,θ)
                return F
            end
            residual = similar(u)
            jac = Float64.(Symbolics.jacobian_sparsity(_f!,residual,u,θ))
            function _f(u)
                return f(u,θ)
            end
            colors = SparseDiffTools.matrix_colors(jac)
            return FiniteDiff.finite_difference_jacobian(_f,u,colorvec=colors,sparsity=jac)
        end

        _prob = NonlinearProblem(NonlinearFunction{false}(f;jac = jac), u0, p)
        sol = solve(_prob, alg.nlsolver; kwargs...)

        return MixedComplementaritySolution(sol.u, sol.resid, prob, alg, sol.retcode)
    end
end
