sd = SymbolicsSparsityDetection()
adtype = AutoSparseFiniteDiff()

for method in (:minmax, :smooth)
    algType = NonlinearReformulation{method}
    op = Symbol("$(method)_transform")
    @eval function __solve(prob::MCP{true}, alg::$algType, u0, p; kwargs...)
        
        function f!(residual, u, θ)
            prob.f(residual, u, θ)
            residual .= $(op).(residual, u, prob.lb, prob.ub)
            return residual
        end

        function _f!(F,u)
            f!(F,u,p)
            return F
        end

        residual = similar(u0)
        #=J0= Float64.(Symbolics.jacobian_sparsity(f!,residual,u0,p))
        colors = SparseDiffTools.matrix_colors(J0)=#
        cache = sparse_jacobian_cache(adtype, sd, _f!,residual, u0)

        function jac!(J,u,θ)
            function _f!(F,u)
                f!(F,u,θ)
                return F
            end
            sparse_jacobian!(J,adtype,cache,_f!,residual,u)
        end

        _prob = NonlinearProblem(NonlinearFunction{true}(f!;jac=jac!), u0, p)
        sol = solve(_prob, alg.nlsolver; kwargs...)

        return MixedComplementaritySolution(sol.u, sol.resid, prob, alg, sol.retcode)
    end

    @eval function __solve(prob::MCP{false}, alg::$algType, u0, p; kwargs...)
        f(u, θ) = $(op).(prob.f(u, θ), u, prob.lb, prob.ub)
        function _f(u)
            return f(u,p)
        end
        
        #residual = similar(u0)
        #J0 = Float64.(Symbolics.jacobian_sparsity(_f!,residual,u0,p))
        #colors = SparseDiffTools.matrix_colors(J0)
        cache = sparse_jacobian_cache(adtype, sd, _f,u0)
        
        function jac(u,θ)   
            function _f(u)
                return f(u,θ)
            end
            return sparse_jacobian(adtype,cache,_f,u)
        end

        _prob = NonlinearProblem(NonlinearFunction{false}(f;jac = jac), u0, p)
        sol = solve(_prob, alg.nlsolver; kwargs...)

        return MixedComplementaritySolution(sol.u, sol.resid, prob, alg, sol.retcode)
    end
end
