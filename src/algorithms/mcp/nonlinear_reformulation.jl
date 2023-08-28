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
        sd_mcp = alg.diffmode isa SparseDiffTools.AbstractSparseADType ? SymbolicsSparsityDetection() : NoSparsityDetection()
        cache = sparse_jacobian_cache(alg.diffmode, sd_mcp, _f!,residual, u0)

        jac_prototype = SparseDiffTools.__init_𝒥(cache)

        function jac!(J,u,p)
            sparse_jacobian!(J,alg.diffmode,cache,_f!,residual,u)
            return J
        end

        _prob = NonlinearProblem(NonlinearFunction{true}(f!;jac=jac!,jac_prototype), u0, p)
        sol = solve(_prob, alg.nlsolver; kwargs...)

        return MixedComplementaritySolution(sol.u, sol.resid, prob, alg, sol.retcode)
    end

    @eval function __solve(prob::MCP{false}, alg::$algType, u0, p; kwargs...)
        f(u, θ) = $(op).(prob.f(u, θ), u, prob.lb, prob.ub)
        function _f(u)
            return f(u,p)
        end
        
        sd_mcp = alg.diffmode isa SparseDiffTools.AbstractSparseADType ? SymbolicsSparsityDetection() : NoSparsityDetection()
        cache = sparse_jacobian_cache(alg.diffmode, sd_mcp, _f,u0)

        jac_prototype = SparseDiffTools.__init_𝒥(cache)
        
        function jac(u,p)   
            return sparse_jacobian(alg.diffmode,cache,_f,u)
        end

        _prob = NonlinearProblem(NonlinearFunction{false}(f;jac = jac,jac_prototype), u0, p)
        sol = solve(_prob, alg.nlsolver; kwargs...)

        return MixedComplementaritySolution(sol.u, sol.resid, prob, alg, sol.retcode)
    end
end
